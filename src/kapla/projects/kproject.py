from __future__ import annotations

import shutil
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from anyio import create_task_group
from pydantic import ValidationError

from kapla.specs.common import BuildSystem
from kapla.specs.kproject import KProjectSpec
from kapla.specs.pyproject import (
    DEFAULT_BUILD_SYSTEM,
    Dependency,
    DependencyMeta,
    Group,
    PoetryConfig,
    PyProjectSpec,
)
from kapla.wrappers.git import GitInfos

from ..core.cmd import Command, get_deadline
from ..core.errors import CommandFailedError
from ..core.finder import find_dirs, find_files
from ..core.io import read_yaml, write_toml, write_yaml
from ..core.logger import logger
from ..core.templates import render_template
from ..specs.kproject import DockerImageSpec, DockerSpec
from .base import BasePythonProject
from .pyproject import KPyProject

if TYPE_CHECKING:
    from .krepo import KRepo


class ReadWriteYAMLMixin:
    _raw: Any

    def read(self, path: Union[str, Path]) -> Any:
        """Read YAML project specs"""
        return read_yaml(path)

    def write(self, path: Union[str, Path]) -> Path:
        """Write YAML project specs"""
        return write_yaml(self._raw, path)


class KProject(ReadWriteYAMLMixin, BasePythonProject[KProjectSpec], spec=KProjectSpec):
    """Base class for pyproject files implementing read and write operations"""

    def __init__(
        self,
        filepath: Union[str, Path],
        repo: Optional[KRepo] = None,
        workspace: Optional[str] = None,
        venv_path: Union[str, Path, None] = None,
    ) -> None:
        super().__init__(filepath, venv_path=repo.venv_path if repo else venv_path)
        self.repo = repo
        self.workspace = workspace
        if self.spec.version is None and self.repo is not None:
            self.spec.version = self.repo.version

    @property
    def gitignore(self) -> List[str]:
        """Constant value at the moment. We should parse project gitignore in the future.

        FIXME: Support reading gitignore from project
        """
        if self.repo:
            return self.repo.gitignore
        else:
            return super().gitignore

    @property
    def pyproject_path(self) -> Path:
        """Path used to write pyproject file"""
        return self.root / "pyproject.toml"

    @property
    def name(self) -> str:
        """The project name"""
        return self.spec.name

    @property
    def slug(self) -> str:
        return self.spec.name.replace("-", "_")

    @property
    def version(self) -> str:
        """The project version"""
        if self.spec.version:
            return self.spec.version
        if self.repo:
            return self.repo.version
        return ""

    def is_already_installed(self) -> bool:
        if self.repo:
            for _ in find_files(
                f"{self.name.replace('-', '_').lower()}.pth",
                root=self.venv_path,
                ignore=[
                    "bin/",
                    "Scripts/",
                    "include",
                    "Include",
                    "share",
                    "doc",
                    "etc",
                ],
            ):
                return True
        return False

    def _extract_dep_names(
        self, dep: Union[str, Dict[str, DependencyMeta]]
    ) -> List[str]:
        """Extract dependency names from a dependency specification"""
        return [dep] if isinstance(dep, str) else list(dep.keys())

    def _collect_deps_from_list(
        self, deps_list: List[Union[str, Dict[str, DependencyMeta]]]
    ) -> Set[str]:
        """Collect dependency names from a list of dependencies"""
        names: Set[str] = set()
        for dep in deps_list:
            names.update(self._extract_dep_names(dep))
        return names

    def get_dependencies_names(self, include_extras: bool = True) -> List[str]:
        """Get a list of dependencies names"""
        names = self._collect_deps_from_list(self.spec.dependencies)

        if include_extras:
            for extra_deps in self.spec.extras.values():
                names.update(self._collect_deps_from_list(extra_deps))

        return list(names)

    def get_local_dependencies_names(self) -> List[str]:
        """Get local dependencies names (include all groups)"""
        return list(self.get_local_dependencies())

    def get_local_dependencies(self) -> Dict[str, Dependency]:
        """Get a dict holding local dependencies of project"""
        # We cannot do anything without a repo
        if self.repo is None:
            return {}

        # Fetch project local dependencies
        local_projects = {
            name: self.repo.projects[name]
            for name in self.get_dependencies_names()
            if name in self.repo.projects
        }
        _need_to_inspect = set(local_projects)
        _inspected: Set[str] = {self.name}
        # For each local dependency
        while _need_to_inspect:
            local_dep = local_projects[_need_to_inspect.pop()]
            for dep in local_dep.get_local_dependencies():
                local_projects[dep] = self.repo.projects[dep]
                if dep not in _inspected:
                    _need_to_inspect.add(dep)
                else:
                    _inspected.add(dep)
        # Gather local dependencies to override
        return {
            name: Dependency.parse_obj({"version": "*"})
            for name, project in local_projects.items()
        }

    def get_build_dependencies(
        self,
        include_local: bool = True,
        include_python: bool = True,
        lock_versions: bool = True,
        process_secondary_dependencies: bool = False,
    ) -> Tuple[Dict[str, Dependency], Dict[str, List[str]], Dict[str, Group]]:
        """Return dependencies, extras and groups"""
        dependencies: Dict[str, Dependency] = {}
        groups: Dict[str, Group] = {}
        extras: Dict[str, List[str]] = {}

        constraints: Dict[str, str]
        if self.repo is None:
            constraints = defaultdict(lambda: "*")
        else:
            constraints = self.repo.get_packages_constraints()

        self._process_main_dependencies(
            dependencies, constraints, lock_versions, process_secondary_dependencies
        )
        self._process_extras_and_groups(
            dependencies,
            extras,
            groups,
            constraints,
            lock_versions,
            process_secondary_dependencies,
        )
        self._handle_python_dependency(dependencies, include_python)
        self._remove_local_dependencies(dependencies, include_local)
        return dependencies, extras, groups

    def _process_main_dependencies(
        self,
        dependencies: Dict[str, Dependency],
        constraints: Union[DefaultDict[str, str], Dict[str, str]],
        lock_versions: bool,
        process_secondary_dependencies: bool,
    ) -> None:
        for dep in self.spec.dependencies:
            self._lock_and_store_dependencies(
                lock_versions, dependencies, constraints, dep
            )
            local_dependencies = self._extract_dep_names(dep)
            if process_secondary_dependencies:
                self._process_secondary_dependencies(
                    local_dependencies, dependencies, constraints, lock_versions
                )

    def _process_secondary_dependencies(
        self,
        local_dependencies: List[str],
        dependencies: Dict[str, Dependency],
        constraints: Union[DefaultDict[str, str], Dict[str, str]],
        lock_versions: bool,
    ) -> None:
        """Process secondary dependencies of the given local dependencies."""
        if not self.repo:
            return

        for dep_name in local_dependencies:
            locked_package = self.repo.packages_lock.packages.get(dep_name)
            if locked_package is None or locked_package.dependencies is None:
                continue
            visited: Set[str] = set()
            for (
                secondary_dep,
                dep_meta_or_version,
            ) in locked_package.dependencies.items():
                self._process_dependency_tree(
                    secondary_dep,
                    visited,
                    dependencies,
                    constraints,
                    lock_versions,
                    self._create_dependency_meta(dep_meta_or_version),
                )

    def _create_dependency_meta(
        self, dep_meta_or_version: Union[str, Dict[str, Any]]
    ) -> None | DependencyMeta:
        return (
            None
            if isinstance(dep_meta_or_version, str)
            else DependencyMeta(**dep_meta_or_version)
        )

    def _process_dependency_tree(
        self,
        dep_name: str,
        visited: Set[str],
        dependencies: Dict[str, Dependency],
        constraints: Union[DefaultDict[str, str], Dict[str, str]],
        lock_versions: bool,
        dep_meta: Optional[DependencyMeta] = None,
    ) -> None:
        """Process a dependency and its subdependencies recursively."""
        if dep_name in visited:
            return
        visited.add(dep_name)

        dep = dep_name if dep_meta is None else {dep_name: dep_meta}

        self._lock_and_store_dependencies(lock_versions, dependencies, constraints, dep)

        if not self.repo:
            return

        locked_package = self.repo.packages_lock.packages.get(dep_name)
        if locked_package is None or locked_package.dependencies is None:
            return

        for (
            sub_dep_name,
            sub_dep_meta_or_version,
        ) in locked_package.dependencies.items():
            self._process_dependency_tree(
                sub_dep_name,
                visited,
                dependencies,
                constraints,
                lock_versions,
                self._create_dependency_meta(sub_dep_meta_or_version),
            )

    def _process_extras_and_groups(
        self,
        dependencies: Dict[str, Dependency],
        extras: Dict[str, List[str]],
        groups: Dict[str, Group],
        constraints: Union[DefaultDict[str, str], Dict[str, str]],
        lock_versions: bool,
        process_secondary_dependencies: bool,
    ) -> None:
        for group_name, group_dependencies in self.spec.extras.items():
            groups[group_name] = Group(dependencies={})
            extras[group_name] = []
            # A dependency can get added to dev group while still being in the in the main dependencies
            # example: if it's a dependency of a dev dependency, or if it has been listed both as a main
            # and a dev dependency by mistake. In that case, it will not be installed with the main dependencies.
            # We add all the main dependencies to the visited set to avoid adding them again
            visited: Set[str] = set(self.get_dependencies_names(include_extras=False))
            for dep in group_dependencies:
                dep_items = [(dep, None)] if isinstance(dep, str) else dep.items()
                for dep_name, value in dep_items:
                    self._add_dependency_to_group(
                        dep_name,
                        group_name,
                        dependencies,
                        extras,
                        groups,
                        constraints,
                        lock_versions,
                        process_secondary_dependencies,
                        visited,
                        value,
                    )

    def _add_dependency_to_group(
        self,
        dep_name: str,
        group_name: str,
        dependencies: Dict[str, Dependency],
        extras: Dict[str, List[str]],
        groups: Dict[str, Group],
        constraints: Union[DefaultDict[str, str], Dict[str, str]],
        lock_versions: bool,
        process_secondary_dependencies: bool,
        visited: Optional[Set[str]] = None,
        value: Optional[DependencyMeta] = None,
    ) -> None:
        if visited is None:
            visited = set()
        if dep_name in visited:
            return
        visited.add(dep_name)
        locked_version = (
            self.get_locked_version(dep_name)
            if lock_versions
            else constraints.get(dep_name, "*")
        )
        dep_obj = (
            Dependency.parse_obj(
                {
                    **value.dict(exclude_unset=True, by_alias=True),
                    "version": locked_version,
                }
            )
            if value
            else Dependency(version=locked_version)
        )
        groups[group_name].dependencies[dep_name] = dep_obj
        if dep_name not in extras[group_name]:
            extras[group_name].append(dep_name)
        if dep_name not in dependencies:
            dep_dict = (
                {
                    **value.dict(exclude_unset=True, by_alias=True),
                    "version": locked_version,
                    "optional": True,
                }
                if value
                else {"version": locked_version, "optional": True}
            )
            dependencies[dep_name] = Dependency.parse_obj(dep_dict)
        if not process_secondary_dependencies:
            return

        if self.repo is None:
            return

        locked_package = self.repo.packages_lock.packages.get(dep_name)
        if not locked_package or not locked_package.dependencies:
            return

        for (
            sub_dep_name,
            sub_dep_meta_or_version,
        ) in locked_package.dependencies.items():
            self._add_dependency_to_group(
                sub_dep_name,
                group_name,
                dependencies,
                extras,
                groups,
                constraints,
                lock_versions,
                process_secondary_dependencies,
                visited,
                self._create_dependency_meta(sub_dep_meta_or_version),
            )

    def _handle_python_dependency(
        self,
        dependencies: Dict[str, Dependency],
        include_python: bool,
    ) -> None:
        if include_python:
            if "python" not in dependencies and self.repo:
                python_dep = self.repo.get_dependency("python")
                if python_dep:
                    dependencies["python"] = python_dep.copy()
        else:
            dependencies.pop("python", None)

    def _remove_local_dependencies(
        self,
        dependencies: Dict[str, Dependency],
        include_local: bool,
    ) -> None:
        if not include_local:
            for dep in self.get_local_dependencies_names():
                dependencies.pop(dep, None)

    def _lock_and_store_dependencies(
        self,
        lock_versions: bool,
        dependencies: Dict[str, Dependency],
        constraints: Union[DefaultDict[str, str], Dict[str, str]],
        dep: Union[str, Dict[str, DependencyMeta]],
    ) -> None:
        if isinstance(dep, str):
            if lock_versions:
                locked_version = self.get_locked_version(dep)
            else:
                locked_version = constraints.get(dep, "*")

            dependencies[dep] = Dependency(version=locked_version)
        else:
            for key, value in dep.items():
                if lock_versions:
                    locked_version = self.get_locked_version(key)
                else:
                    locked_version = constraints.get(key, "*")
                dependencies[key] = Dependency.parse_obj(
                    {
                        **value.dict(exclude_unset=True, by_alias=True),
                        "version": locked_version,
                    }
                )

    def get_locked_version(self, package: str) -> str:
        if self.repo:
            return self.repo.get_locked_version(package)
        else:
            return "*"

    def get_pyproject_spec(
        self,
        lock_versions: bool = True,
        build_system: BuildSystem = DEFAULT_BUILD_SYSTEM,
        process_secondary_dependencies: bool = False,
    ) -> PyProjectSpec:
        """Create content of pyproject.toml file according to project.yaml"""
        dependencies, extras, groups = self.get_build_dependencies(
            lock_versions=lock_versions,
            process_secondary_dependencies=process_secondary_dependencies,
        )
        # Gather raw tool.poetry configuration byt exclude dependencies, extras and group fields
        raw_poetry_config = self.spec.dict(
            by_alias=True,
            exclude_unset=True,
            exclude={"dependencies", "extras", "group", "docker"},
        )
        # Generate poetry config by merging raw config and gather dependencies, extras and group
        poetry_config = PoetryConfig(
            **raw_poetry_config,
            dependencies=dependencies,  # pyright: ignore
            extras=extras,
            group=groups,
        )
        # Generate pyproject file
        return PyProjectSpec(
            tool={"poetry": poetry_config}, build_system=build_system  # pyright: ignore
        )

    def write_pyproject(
        self,
        path: Union[str, Path, None] = None,
        lock_versions: bool = True,
        build_system: BuildSystem = DEFAULT_BUILD_SYSTEM,
        process_secondary_dependencies: bool = False,
    ) -> KPyProject:
        """Write auto-generated pyproject.toml file.

        If path argument is not specified, file is generated in the project directory by default.
        """
        spec = self.get_pyproject_spec(
            lock_versions=lock_versions,
            build_system=build_system,
            process_secondary_dependencies=process_secondary_dependencies,
        )
        pyproject_path = Path(path) if path else self.pyproject_path
        content = spec.dict()
        # Create an inline table to have more readable pyprojects
        if spec.tool.poetry.dependencies:
            content["tool"]["poetry"]["dependencies"] = (
                KPyProject._create_inline_tables(
                    content["tool"]["poetry"]["dependencies"]
                )
            )
        # Ensure python dependency is a string
        if "python" in content["tool"]["poetry"]["dependencies"]:
            content["tool"]["poetry"]["dependencies"]["python"] = content["tool"][
                "poetry"
            ]["dependencies"]["python"]["version"]
        # Write pyproject.toml as file
        write_toml(content, pyproject_path)
        try:
            # Parse pyproject we just wrote so that we're sure it is valid
            pyproject = KPyProject(pyproject_path, repo=self.repo)
        except ValidationError as err:
            logger.error(
                "Failed to validate pyproject",
                exc_info=err,
                path=pyproject_path.as_posix(),
            )
            raise
        return pyproject

    def remove_pyproject(self, pyproject_path: Union[str, Path, None] = None) -> None:
        """Remove auto-generated poetry files"""
        pyproject_path = Path(pyproject_path) if pyproject_path else self.pyproject_path
        pyproject_path.unlink(missing_ok=True)
        lock_path = pyproject_path.parent / "poetry.lock"
        lock_path.unlink(missing_ok=True)

    @contextmanager
    def temporary_pyproject(
        self,
        path: Union[str, Path, None] = None,
        lock_versions: bool = True,
        build_system: BuildSystem = DEFAULT_BUILD_SYSTEM,
        clean: bool = True,
        process_secondary_dependencies: bool = False,
    ) -> Iterator[KPyProject]:
        """A context manager which ensures pyproject.toml is written to disk within context and removed out of context"""
        pyproject = self.write_pyproject(
            path,
            lock_versions=lock_versions,
            build_system=build_system,
            process_secondary_dependencies=process_secondary_dependencies,
        )
        try:
            yield pyproject
        finally:
            if clean:
                self.remove_pyproject()

    async def build(
        self,
        env: Optional[Mapping[str, Any]] = None,
        build_system: BuildSystem = DEFAULT_BUILD_SYSTEM,
        lock_versions: bool = True,
        clear_dist: bool = True,
        clean: bool = True,
        quiet: bool = False,
        raise_on_error: bool = False,
        timeout: Optional[float] = None,
        deadline: Optional[float] = None,
        recurse: bool = True,
        process_secondary_dependencies: bool = False,
        **kwargs: Any,
    ) -> Command:
        if recurse and self.repo:
            async with create_task_group() as tg:
                for name in self.get_local_dependencies_names():
                    tg.start_soon(
                        partial(
                            self.repo.projects[name].build,
                            env=env,
                            quiet=quiet,
                            build_system=build_system,
                            lock_versions=lock_versions,
                            recurse=False,
                            process_secondary_dependencies=process_secondary_dependencies,
                        )
                    )
        if clear_dist:
            shutil.rmtree(self.root / "dist", ignore_errors=True)
        with self.temporary_pyproject(
            self.pyproject_path,
            lock_versions=lock_versions,
            build_system=build_system,
            clean=clean,
            process_secondary_dependencies=process_secondary_dependencies,
        ) as pyproject:
            return await pyproject.poetry_build(
                env=env,
                quiet=quiet,
                raise_on_error=raise_on_error,
                timeout=timeout,
                deadline=deadline,
                **kwargs,
            )

    async def install(
        self,
        exclude_groups: Union[Iterable[str], str, None] = None,
        include_groups: Union[Iterable[str], str, None] = None,
        only_groups: Union[Iterable[str], str, None] = None,
        default: bool = False,
        lock_versions: bool = True,
        force: bool = False,
        build_system: BuildSystem = DEFAULT_BUILD_SYSTEM,
        build_isolation: bool = True,
        clean: bool = True,
        quiet: bool = False,
        raise_on_error: bool = False,
        timeout: Optional[float] = None,
        deadline: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Command]:
        if not self.repo:
            raise NotImplementedError(
                "PEP 660 install is not supported without parent repo"
            )
        if self.is_already_installed():
            if not force:
                return None
        groups = list(self.spec.extras)
        if default:
            groups = []
        elif only_groups:
            groups = [group for group in groups if group in only_groups]
        else:
            if exclude_groups:
                groups = [group for group in groups if group not in exclude_groups]
            if include_groups:
                groups = [group for group in groups if group in include_groups]
        target = self.root.as_posix()
        if groups:
            target += f'[{",".join(groups)}]'
        cmd = ["-e", target]
        if not build_isolation:
            cmd.append("--no-build-isolation")
        cmd.append("--no-deps")
        logger.debug(
            f"Installing {self.name} with command: {cmd}",
            version=self.version,
            package=target,
        )
        with self.temporary_pyproject(
            self.pyproject_path,
            clean=clean,
            lock_versions=lock_versions,
            build_system=build_system,
        ):

            return await self.repo.pip_install(
                *cmd,
                quiet=quiet,
                raise_on_error=raise_on_error,
                timeout=timeout,
                deadline=deadline,
                **kwargs,
            )

    async def add_dependency(
        self,
        package: str,
        group: Optional[str] = None,
        editable: bool = False,
        extras: Union[str, List[str], None] = None,
        optional: bool = False,
        python: Optional[str] = None,
        platform: Union[str, List[str], None] = None,
        source: Optional[str] = None,
        allow_prereleases: bool = False,
        dry_run: bool = False,
        lock: bool = False,
        quiet: bool = False,
        raise_on_error: bool = False,
        timeout: Optional[float] = None,
        deadline: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Union[str, Dependency]]:
        if group:
            repo_group = self.name + "--" + group
        else:
            repo_group = self.name
        if self.repo:
            group_before = self.repo.spec.tool.poetry.group.get(repo_group, Group())
            await self.repo.poetry_add(
                package,
                group=repo_group,
                editable=editable,
                extras=extras,
                optional=optional,
                python=python,
                platform=platform,
                source=source,
                allow_prereleases=allow_prereleases,
                dry_run=dry_run,
                lock=lock,
                quiet=quiet,
                raise_on_error=raise_on_error,
                timeout=timeout,
                deadline=deadline,
                **kwargs,
            )
            # Refresh repo metadata
            self.repo.refresh()
            group_after = self.repo.spec.tool.poetry.group[repo_group]
            # Get package diff
            new_packages = set(group_after.dependencies).difference(
                group_before.dependencies
            )
            # Add new packages to project.yml raw spec
            if group is None:
                for package in new_packages:
                    self._raw["dependencies"].append(package)
            else:
                if group not in self._raw["extras"]:
                    self._raw["extras"][group] = []
                for package in new_packages:
                    self._raw["extras"][group].append(package)
            # Write spec
            self.write(self.root / "project.yml")
            self.refresh()
            # Return new packages
            return {name: group_after.dependencies[name] for name in new_packages}

        else:
            raise NotImplementedError(
                "It's not possible to add dependencies to projet without parent repo"
            )

    async def remove_dependency(
        self,
        package: str,
        group: Optional[str] = None,
        dry_run: bool = False,
        quiet: bool = False,
        raise_on_error: bool = False,
        timeout: Optional[float] = None,
        deadline: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Union[str, Dependency]]]:
        if group:
            repo_group = self.name + "--" + group
        else:
            repo_group = self.name
        if self.repo:
            group_before = self.repo.spec.tool.poetry.group.get(repo_group, Group())
            try:
                await self.repo.poetry_remove(
                    package,
                    group=repo_group,
                    dry_run=dry_run,
                    quiet=quiet,
                    raise_on_error=raise_on_error,
                    timeout=timeout,
                    deadline=deadline,
                    **kwargs,
                )
            except CommandFailedError:
                # Try to remove dep anyway
                return None
            # Refresh repo metadata
            self.repo.refresh()
            group_after = self.repo.spec.tool.poetry.group[repo_group]
            # Get package diff
            removed_packages = set(group_before.dependencies).difference(
                group_after.dependencies
            )
            # Remove package from project.yml
            if group is None:
                self._raw["dependencies"] = [
                    dep
                    for dep in self._raw["dependencies"]
                    if dep not in removed_packages
                ]
            else:
                if group in self._raw["extras"]:
                    self._raw["extras"][group] = [
                        dep
                        for dep in self._raw["dependencies"][group]
                        if dep not in removed_packages
                    ]
            # Write and refresh
            if removed_packages:
                self.write(self.root / "project.yml")
                self.refresh()
            # Return removed packages
            return {name: group_before.dependencies[name] for name in removed_packages}

        else:
            raise NotImplementedError(
                "It's not possible to remove dependencies to projet without parent repo"
            )

    async def get_docker_tag(self, git_infos: Optional[GitInfos] = None) -> str:
        # Gather git infos
        git_infos = git_infos or await self.get_git_infos()
        # Gather tag
        # Check if it's a release
        if git_infos.tag:
            tag = git_infos.tag
        # Use branch and commit if available
        elif git_infos.branch and git_infos.commit:
            tag = "-".join([git_infos.branch.split("/")[-1].lower(), git_infos.commit])
        # Use commit only
        elif git_infos.commit:
            tag = git_infos.commit
        else:
            tag = "latest"
        return tag

    async def build_docker(
        self,
        tag: Optional[str] = None,
        additional_tags: Optional[Iterable[str]] = None,
        load: bool = False,
        push: bool = False,
        build_args: Optional[Dict[str, str]] = None,
        platforms: Optional[List[str]] = None,
        provenance: bool = False,
        suffix: Optional[str] = None,
        output_dir: Union[str, Path, None] = None,
        build_dist: bool = True,
        lock_versions: bool = True,
        build_dist_env: Optional[Dict[str, str]] = None,
        build_dist_system: BuildSystem = DEFAULT_BUILD_SYSTEM,
        quiet: bool = False,
        raise_on_error: bool = False,
        timeout: Optional[float] = None,
        deadline: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Command]:
        """Build docker image for the project according to the project spec."""
        suffix = suffix or ""
        cmds: List[Command] = []
        deadline = get_deadline(timeout, deadline)
        kwargs["rc"] = kwargs.get("rc", 0 if raise_on_error else None)
        spec = self.spec.docker
        if not spec:
            raise ValueError("No docker spec found for project")
        if not self.repo:
            raise ValueError("Cannot build docker images without parent repo")
        git_infos = await self.get_git_infos()
        images = self._get_images(spec)

        for image in images:
            logger.info(f"Building image {image.name}, template {image.template}")
            tag = tag or await self.get_docker_tag(git_infos)
            self._render_template(image, spec)
            try:
                cmd = self._create_docker_command(
                    image,
                    tag,
                    suffix,
                    spec,
                    git_infos,
                    build_args,
                    deadline,
                    quiet,
                    load,
                    push,
                    output_dir,
                    platforms,
                    provenance,
                    additional_tags,
                    **kwargs,
                )
                if build_dist:
                    await self._build_dist(
                        build_dist_env,
                        build_dist_system,
                        lock_versions,
                        deadline,
                        True,
                        **kwargs,
                    )
                logger.info("Invoking docker command", command=cmd.cmd)
                cmds.append(await cmd.run())
            finally:
                Path(self.root, "Dockerfile").unlink(missing_ok=True)
        return cmds

    def _get_images(self, spec: DockerSpec) -> List[DockerImageSpec]:
        if spec.images is not None:
            return spec.images
        elif spec.image is not None:
            return [DockerImageSpec(name=spec.image, template=spec.template)]
        else:
            return []

    def _render_template(self, image: DockerImageSpec, spec: DockerSpec) -> None:
        template = image.template or "library"
        template_args = spec.options or {}
        template_file = "Dockerfile." + template
        assert self.repo
        template_path = self.repo.root / ".repo/templates/dockerfiles" / template_file

        render_template(template_path, self.root / "Dockerfile", **template_args)

    def _create_docker_command(
        self,
        image: DockerImageSpec,
        tag: str,
        suffix: str,
        spec: DockerSpec,
        git_infos: GitInfos,
        build_args: Optional[Dict[str, str]],
        deadline: Optional[float],
        quiet: bool,
        load: bool,
        push: bool,
        output_dir: Union[str, Path, None],
        platforms: Optional[List[str]],
        provenance: bool,
        additional_tags: Optional[Iterable[str]],
        **kwargs: Any,
    ) -> Command:
        cmd = Command("docker buildx build", deadline=deadline, quiet=quiet, **kwargs)
        build_args = self._prepare_build_args(spec, image, git_infos, build_args, tag)
        # add specific build arg for this image to build
        logger.warning("Using build args", build_args=build_args)
        for key, value in build_args.items():
            cmd.add_option("--build-arg", "=".join([key, value]), escape=True)
        cmd.add_repeat_option("--label", spec.labels)
        cmd.add_repeat_option(
            "--label",
            [
                f"quara.package.version={self.version}",
                f"quara.package.name={self.name}",
            ],
        )
        self._add_git_labels(cmd, git_infos)
        cmd.add_option("--tag", ":".join([image.name + suffix, tag]))
        self._add_additional_tags(cmd, image, suffix, additional_tags)
        self._add_optional_options(
            cmd, load, push, output_dir, platforms, provenance, spec
        )
        cmd.add_argument(
            Path(self.root, spec.context).resolve(True).as_posix()
            if spec.context
            else self.root.as_posix()
        )
        return cmd

    def _prepare_build_args(
        self,
        spec: DockerSpec,
        image: DockerImageSpec,
        git_infos: GitInfos,
        build_args: Optional[Dict[str, str]],
        tag: str,
    ) -> Dict[str, str]:
        _build_args = spec.build_args.copy() if spec.build_args else {}
        if build_args:
            _build_args.update(build_args)
        build_args = _build_args.copy()
        # add specific build args for this image
        if image.build_args is not None:
            build_args.update(image.build_args)
        if spec.base_image and "BASE_IMAGE" not in build_args:
            base_image = (
                spec.base_image + ":" + tag
                if ":" not in spec.base_image
                else spec.base_image
            )
            build_args["BASE_IMAGE"] = base_image
        build_args["PACKAGE_NAME"] = self.name
        build_args["PACKAGE_VERSION"] = self.version
        if git_infos.commit:
            build_args["GIT_COMMIT"] = git_infos.commit
        if git_infos.branch:
            build_args["GIT_BRANCH"] = git_infos.branch
        if git_infos.tag:
            build_args["GIT_TAG"] = git_infos.tag
        return build_args

    def _add_git_labels(self, cmd: Command, git_infos: GitInfos) -> None:
        if git_infos.tag:
            cmd.add_option("--label", f"git.tag.name={git_infos.tag}")
        if git_infos.branch:
            cmd.add_option("--label", f"git.branch.name={git_infos.branch}")
        if git_infos.commit:
            cmd.add_option("--label", f"git.commit={git_infos.commit}")

    def _add_additional_tags(
        self,
        cmd: Command,
        image: DockerImageSpec,
        suffix: str,
        additional_tags: Optional[Iterable[str]],
    ) -> None:
        for tag in additional_tags or []:
            cmd.add_option("--tag", ":".join([image.name + suffix, tag]))

    def _add_optional_options(
        self,
        cmd: Command,
        load: bool,
        push: bool,
        output_dir: Union[str, Path, None],
        platforms: Optional[List[str]],
        provenance: bool,
        spec: DockerSpec,
    ) -> None:
        if load:
            cmd.add_option("--load")
        if push:
            cmd.add_option("--push")
        if output_dir is not None:
            cmd.add_option(
                "--output", "type=local,dest=" + Path(self.root, output_dir).as_posix()
            )
        cmd.add_option(
            "--metadata-file",
            Path(
                self.root,
                "dist",
                "-".join([self.name, self.version]) + ".docker-metadata",
            ).as_posix(),
        )
        platform = platforms if platforms else spec.platforms
        if platform:
            cmd.add_repeat_option("--platform", platform)
        if provenance is False:
            cmd.add_option("--provenance", "false")

    async def _build_dist(
        self,
        build_dist_env: Optional[Dict[str, str]],
        build_dist_system: BuildSystem,
        lock_versions: bool,
        deadline: Optional[float],
        process_secondary_dependencies: bool,
        **kwargs: Any,
    ) -> None:
        await self.build(
            env=build_dist_env,
            build_system=build_dist_system,
            lock_versions=lock_versions,
            quiet=True,
            deadline=deadline,
            process_secondary_dependencies=process_secondary_dependencies,
            **kwargs,
        )
        dist_root = self.root / "dist"
        dist_root.mkdir(exist_ok=True, parents=False)
        assert self.repo
        for dep in self.repo.list_projects(include=[self.name]):
            if dep.name == self.name:
                continue
            for filepath in Path(dep.root, "dist").glob("*.whl"):
                shutil.copy2(filepath, dist_root)

    def clean(self) -> None:
        """Remove well-known non versioned files"""
        # Remove venv
        shutil.rmtree(self.venv_path, ignore_errors=True)
        # Remove directories
        for path in find_dirs(
            self.gitignore,
            self.root,
        ):
            shutil.rmtree(path, ignore_errors=True)
        # Remove files
        for path in find_files(
            self.gitignore,
            root=self.root,
        ):
            path.unlink(missing_ok=True)
