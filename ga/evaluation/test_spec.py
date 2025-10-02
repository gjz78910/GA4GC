from __future__ import annotations

import hashlib
import json
import platform
import re

from dataclasses import dataclass
from typing import Any, Union, cast

from .constants import (
    SWEPerfInstance,
    KEY_INSTANCE_ID,
    FAIL_TO_PASS,
    PASS_TO_PASS,
    MAP_REPO_TO_INSTALL,
    MAP_REPO_VERSION_TO_SPECS,
    USE_X86,
)
from .dockerfiles import (
    get_dockerfile_base,
    get_dockerfile_env,
    get_dockerfile_instance,
)
from .utils import (
    get_requirements,
    get_environment_yml,
    get_test_directives,
)
from tqdm import tqdm

DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"
REPEAT_TIME = 20 # TODO

@dataclass
class TestSpec:
    """
    A dataclass that represents a test specification for a single instance of SWE-Perf.
    """
    instance_id: str
    base_commit: str
    repo: str
    version: str
    repo_script_list: list[str]
    eval_script_list: list[str]
    eval_script_list_alltests: list[str]
    eval_script_list_coverage: list[str]
    eval_script_list_function: list[str]
    env_script_list: list[str]
    arch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    efficiency_test: list[str]
    is_eval: bool
    patch: str = None
    test_patch: str = None

    @property
    def setup_env_script(self):
        return "\n".join(["#!/bin/bash", "set -exo pipefail"] + self.env_script_list) + "\n"

    @property
    def eval_script(self):
        return "\n".join(["#!/bin/bash", "set -xo pipefail"] + self.eval_script_list) + "\n"
        # Don't exit early because we need to revert tests at the end
    
    @property
    def eval_script_alltests(self):
        return "\n".join(["#!/bin/bash", "set -xo pipefail"] + self.eval_script_list_alltests) + "\n"
        # Don't exit early because we need to revert tests at the end

    @property
    def eval_script_coverage(self):
        """
        Returns the script to run the coverage tests.
        """
        return "\n".join(["#!/bin/bash", "set -exo pipefail"] + self.eval_script_list_coverage) + "\n"
    
    @property
    def eval_script_function(self):
        """
        Returns the script to run the coverage tests.
        """
        return "\n".join(["#!/bin/bash", "set -exo pipefail"] + self.eval_script_list_function) + "\n"

    @property
    def install_repo_script(self):
        return "\n".join(["#!/bin/bash", "set -exo pipefail"] + self.repo_script_list) + "\n"

    @property
    def base_image_key(self):
        return f"base.{self.arch}:latest"

    @property
    def env_image_key(self):
        """
        SUPER-OPTIMIZED: Use the super optimized env image key to maximize image sharing.
        This ensures instances with identical environment specs share the same Docker image.
        """
        return self.super_optimized_env_image_key

    # @property
    # def optimized_env_image_key(self):
    #     """
    #     OPTIMIZED: Environment image key based on repo + version instead of per-instance.
    #     This allows multiple instances from the same repo version to share the same environment image.
    #     """
    #     # Create a hash based on repo, version, and environment specs
    #     hash_object = hashlib.sha256()
    #     hash_object.update(f"{self.repo}:{self.version}".encode("utf-8"))
    #     hash_object.update(str(self.env_script_list).encode("utf-8"))
    #     hash_value = hash_object.hexdigest()
    #     val = hash_value[:16]  # 16 characters for repo-version level
    #     return f"env.{self.repo.replace('/', '__')}.{self.version}.{val}:latest"

    @property
    def super_optimized_env_image_key(self):
        """
        SUPER-OPTIMIZED: Environment image key based on environment configuration, not instance-specific content.
        This approach ensures instances with identical environment specs share the same Docker image,
        regardless of their specific commit requirements.
        
        The key insight is that instances with the same repo, version, Python version, and package type
        should share the same environment image, even if their requirements files differ slightly.
        """
        # Hash based on environment specs, not instance-specific requirements
        hash_object = hashlib.sha256()
        
        # Base configuration: repo, version, architecture
        hash_object.update(f"{self.repo}:{self.version}:{self.arch}".encode("utf-8"))
        
        # Get the specs for this repo/version
        specs = MAP_REPO_VERSION_TO_SPECS[self.repo][self.version]
        
        # Hash the environment configuration (excluding instance-specific content)
        env_config = {
            "python": specs.get("python", ""),
            "packages": specs.get("packages", ""),
            "pip_packages": sorted(specs.get("pip_packages", [])),  # Sort for consistent hashing
            "env_patches": sorted(specs.get("env_patches", []))     # Sort for consistent hashing
        }
        hash_object.update(str(env_config).encode("utf-8"))
        
        hash_value = hash_object.hexdigest()
        val = hash_value[:22]  # 22 characters is still very likely to be unique
        return f"env.{self.arch}.{val}:latest"
    


    @property
    def instance_image_key(self):
        """
        SIMPLIFIED: Instance image key that ensures unique images for each instance.
        This creates separate images for different repositories/commits since they need
        different source code.
        """
        # Create a simple hash based on instance-specific information
        hash_object = hashlib.sha256()
        hash_object.update(f"{self.repo}:{self.base_commit}:{self.instance_id}".encode("utf-8"))
        instance_hash = hash_object.hexdigest()[:16]
        
        # Instance images are different from environment images
        return f"instance.{self.arch}.{instance_hash}:latest"



    def get_instance_container_name(self, run_id=None):
        if not self.is_eval:
            if not run_id:
                return f"{self.repo.replace('/', '__')}.{self.version}"
            return f"{self.repo.replace('/', '__')}.{self.version}.{run_id}"
        else:
            if not run_id:
                return f"{self.instance_id}"
            return f"{self.instance_id.lower()}.{run_id}"

    @property
    def base_dockerfile(self):
        return get_dockerfile_base(self.platform, self.arch)

    @property
    def env_dockerfile(self):
        return get_dockerfile_env(self.platform, self.arch)

    @property
    def instance_dockerfile(self):
        return get_dockerfile_instance(self.platform, self.env_image_key)

    @property
    def platform(self):
        if self.arch == "x86_64":
            return "linux/x86_64"
        elif self.arch == "arm64":
            return "linux/arm64/v8"
        else:
            raise ValueError(f"Invalid architecture: {self.arch}")


def get_test_specs_from_dataset(dataset: Union[list[SWEPerfInstance], list[TestSpec]]) -> list[TestSpec]:
    """
    Idempotent function that converts a list of SWEPerfInstance objects to a list of TestSpec objects.
    """
    if isinstance(dataset[0], TestSpec):
        return cast(list[TestSpec], dataset)
    return list(tqdm(map(make_test_spec, cast(list[SWEPerfInstance], dataset)), total = len(dataset)))


def make_repo_script_list(specs, repo, repo_directory, base_commit, env_name):
    """
    Create a list of bash commands to set up the repository for testing.
    This is the setup script for the instance image.
    """
    setup_commands = [
        f"git clone -o origin https://github.com/{repo} {repo_directory}",
        f"chmod -R 777 {repo_directory}",  # So nonroot user can run tests
        f"cd {repo_directory}",
        f"git fetch origin {base_commit}",
        f"git reset --hard {base_commit}",
        # Remove the remote so the agent won't see newer commits.
        "git remote remove origin",
        # Make sure conda is available for later use
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        'echo "Current environment: $CONDA_DEFAULT_ENV"',
    ]
    if repo in MAP_REPO_TO_INSTALL:
        setup_commands.append(MAP_REPO_TO_INSTALL[repo])

    # Run pre-install set up if provided
    if "pre_install" in specs:
        for pre_install in specs["pre_install"]:
            setup_commands.append(pre_install)

    if "install" in specs:
        setup_commands.append(specs["install"])
    return setup_commands


def replace_uninstallable_packages_requirements_txt(requirement_str: str) -> str:
    """Replaces certain packages in a requirements.txt-like string.
    For example, some packages have been yanked and we need to replace them with compatible alternatives.
    """
    replacements = {
        # See https://github.com/princeton-nlp/SWE-bench/issues/199
        # This package was sinced yanked, so we need to force pip
        # to install it.
        # "types-pkg_resources": "types-pkg-resources==0.1.3",
    }
    requirements = [req.strip() for req in requirement_str.split("\n") if req.strip()]
    requirements_replaced = []
    for requirement in requirements:
        if requirement in replacements:
            print(f"Replaced {requirement!r} with {replacements[requirement]!r} (replace_uninstallable_packages)")
            requirements_replaced.append(replacements[requirement])
        else:
            requirements_replaced.append(requirement)
    return "\n".join(requirements_replaced) + "\n"


def make_env_script_list(instance: SWEPerfInstance, specs: dict, env_name: str) -> list[str]:
    """
    Creates the list of commands to set up the conda environment for testing.
    This is the setup script for the environment image.

    Returns:
        list[str]: List of commands to set up the conda environment
    """
    HEREDOC_DELIMITER = "EOF_59812759871"
    reqs_commands = [
        "source /opt/miniconda3/bin/activate",
    ]
    # Create conda environment according to install instructinos
    pkgs = specs.get("packages", "")
    if pkgs == "requirements.txt":
        # Create environment
        cmd = f"conda create -n {env_name} python={specs['python']} -y"
        reqs_commands.append(cmd)

        # Install dependencies
        reqs = replace_uninstallable_packages_requirements_txt(get_requirements(instance))
        path_to_reqs = "$HOME/requirements.txt"
        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        if "env_patches" in specs:
            reqs_commands += specs["env_patches"]
        cmd = f"conda activate {env_name} && python -m pip install -r {path_to_reqs}"
        reqs_commands.append(cmd)
        reqs_commands.append(f"rm {path_to_reqs}")
    elif pkgs == "environment.yml":
        # Create environment from yml
        reqs = get_environment_yml(instance, env_name)
        if reqs == None: # Can't find yml
            return []
        path_to_reqs = "environment.yml"
        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        if "env_patches" in specs:
            reqs_commands += specs["env_patches"]
        if "no_use_env" in specs and specs["no_use_env"]:
            # `conda create` based installation
            cmd = f"conda create -c conda-forge -n {env_name} python={specs['python']} -y"
            reqs_commands.append(cmd)

            # Install dependencies
            cmd = f"conda env update -f {path_to_reqs}"
            reqs_commands.append(cmd)
        else:
            # `conda env create` based installation
            cmd = f"conda env create --file {path_to_reqs}"
            reqs_commands.append(cmd)

            if 'python' in specs:
                cmd = f"conda activate {env_name} && conda install python={specs['python']} -y"
            else:
                cmd = f"conda activate {env_name}"
            reqs_commands.append(cmd)

        # Remove environment.yml
        reqs_commands.append(f"rm {path_to_reqs}")
    else:
        # Create environment + install dependencies
        if "env_patches" in specs:
            reqs_commands += specs["env_patches"]
        cmd = f"conda create -n {env_name} python={specs['python']} {pkgs} -y"
        reqs_commands.append(cmd)

    reqs_commands.append(f"conda activate {env_name}")

    # Install additional packages if specified
    if "pip_packages" in specs:
        pip_packages = " ".join(specs["pip_packages"])
        cmd = f"python -m pip install {pip_packages}"
        reqs_commands.append(cmd)
    return reqs_commands

def make_test_command(instance):
    # if instance['repo'] == "python/mypy":
    #     pattern = r'\[case ([^\]]+)\]'
    #     test_keys = re.findall(pattern, instance["test_patch"])
    #     test_keys_or = " or ".join(test_keys)
    #     test_command = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"] + " " + f'"{test_keys_or}"'
    #     return test_command
    # else:
    #     test_command = " ".join(
    #         [
    #             MAP_REPO_VERSION_TO_SPECS[instance["repo"].lower()][instance["version"]]["test_cmd"],
    #             *get_test_directives(instance),
    #         ]
    #     )
    #     return test_command
    commands = []
    joined_tests = "\' \'".join(instance['efficiency_test'])
    # warm up the cache
    commands.append(
                    MAP_REPO_VERSION_TO_SPECS[instance["repo"].lower()][instance["version"]]["test_all_cmd"] + f" -vv --json-report --json-report-file=report.json " + f"\'{joined_tests}\'"
    )
    commands.append(
                    MAP_REPO_VERSION_TO_SPECS[instance["repo"].lower()][instance["version"]]["test_all_cmd"] + f" -vv --json-report --json-report-file=report.json " + f"\'{joined_tests}\'"
    )
    commands.append(
                    MAP_REPO_VERSION_TO_SPECS[instance["repo"].lower()][instance["version"]]["test_all_cmd"] + f" -vv --json-report --json-report-file=report.json " + f"\'{joined_tests}\'"
    )
    # for idx, test in enumerate(instance["efficiency_test"]):
    #     command = MAP_REPO_VERSION_TO_SPECS[instance["repo"].lower()][instance["version"]]["test_all_cmd"] + f" -vv --json-report --json-report-file=report{idx}.json " + f"\'{test}\'"
    #     commands.append(command)
    for re_idx in range(REPEAT_TIME):
        commands.append(
            MAP_REPO_VERSION_TO_SPECS[instance["repo"].lower()][instance["version"]]["test_all_cmd"] + f" -vv --json-report --json-report-file=report{re_idx}.json " + f"\'{joined_tests}\'"
        )
    return commands

    
def make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch):
    """
    Applies the test patch and runs the tests.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    # apply_test_patch_command = (
    #     f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    # )
    test_command = make_test_command(instance)
    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command, "pip install pytest-json-report"] + \
        test_command + \
        [reset_tests_command]  # Revert tests after done, leave the repo in the same state as before
    return eval_commands

def make_test_command_alltests(instance):
    if instance['repo'] == "python/mypy":
        pattern = r'\[case ([^\]]+)\]'
        # test_keys = re.findall(pattern, instance["test_patch"])
        # test_keys_or = " or ".join(test_keys)
        test_command = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_all_cmd"] #+ " " + f'"{test_keys_or}"'
        return test_command
    else:
        test_command = " ".join(
            [
                MAP_REPO_VERSION_TO_SPECS[instance["repo"].lower()][instance["version"]]["test_all_cmd"],
                # *get_test_directives(instance),
            ]
        )
        return test_command

def make_eval_script_list_alltests(instance, specs, env_name, repo_directory, base_commit):
    """
    Runs all tests.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    # Reset test files to the state they should be in before the patch.
    test_command = make_test_command_alltests(instance)
    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        test_command,
    ]
    return eval_commands

def make_eval_script_list_coverage(instance, specs, env_name, repo_directory, base_commit, test_patch):
    """
    Applies the test patch and runs the tests.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    # apply_test_patch_command = (
    #     f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    # )
    test_commands = []
    test_commands.append("pip install coverage")
    for idx, test in enumerate(instance["efficiency_test"]):
        test_commands.append(" ".join(
            [
                f"COVERAGE_FILE=.coverage.test{idx} coverage run -m",
                MAP_REPO_VERSION_TO_SPECS[instance["repo"].lower()][instance["version"]]["test_all_cmd"],
                f"\'{test}\'",
            ]
        ))
        test_commands.extend([
            f"mkdir tmp_coverage_test{idx}",
            f"cp .coverage.test{idx}* tmp_coverage_test{idx}/", # some file may have magic string, e.g., .coverage.test0.n121-008-254.3408726.XrXooqWx
            f"cd tmp_coverage_test{idx}",
            f"COVERAGE_FILE=.coverage.test{idx} coverage combine",
            f"cd ../",
            f"COVERAGE_FILE=tmp_coverage_test{idx}/.coverage.test{idx} coverage xml -o /tmp/coverage_test{idx}.xml"
        ]
        )

    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command] + \
        test_commands + \
        [
            reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
    ]
    return eval_commands

def make_eval_script_list_function(instance, specs, env_name, repo_directory, base_commit, test_patch):
    """
    Applies the test patch and runs the tests.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    # apply_test_patch_command = (
    #     f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    # )
    test_commands = []
    test_commands.append("pip install yappi")
    for idx, test in enumerate(instance["efficiency_test"]):
        test_commands.append(" ".join(
            [
                f"python profile_test.py --output function{idx}.json --test ",
                f"\'{test}\'",
                "--",
                MAP_REPO_VERSION_TO_SPECS[instance["repo"].lower()][instance["version"]]["test_all_cmd"],
            ]
        ))

    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command] + \
        test_commands + \
        [
            reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
    ]
    return eval_commands


def make_test_spec(instance: SWEPerfInstance, is_eval = False) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    instance_id = instance[KEY_INSTANCE_ID]
    # if there's capital letters in the repo name, convert to lowercase
    if instance_id != instance_id.lower():
        print(f"Instance ID {instance_id} contains capital letters. Converting to lowercase.")
        instance_id = instance_id.lower()
    repo = instance["repo"].lower()
    version = instance["version"]
    base_commit = instance["base_commit"]
    test_patch = instance["test_patch"]
    if "efficiency_test" not in instance:
        efficiency_test = None
        instance['efficiency_test'] = ""
    else:
        efficiency_test = instance['efficiency_test']
    
    if is_eval:
        patch = instance.get("patch", "")
    else:
        patch = None

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]
    
    if PASS_TO_PASS in instance:
        try:
            pass_to_pass = _from_json_or_obj(PASS_TO_PASS)
        except Exception as e:
            print(f"Error parsing PASS_TO_PASS for instance {instance_id}: {e}. PASS_TO_PASS: {instance[PASS_TO_PASS]}")
            pass_to_pass = []

        try:
            fail_to_pass = _from_json_or_obj(FAIL_TO_PASS)
        except Exception as e:
            print(f"Error parsing FAIL_TO_PASS for instance {instance_id}: {e}. FAIL_TO_PASS: {instance[FAIL_TO_PASS]}")
            fail_to_pass = []
    else:
        pass_to_pass = []
        fail_to_pass = []

    env_name = "testbed"
    repo_directory = f"/{env_name}"
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    repo_script_list = make_repo_script_list(specs, repo, repo_directory, base_commit, env_name)
    env_script_list = make_env_script_list(instance, specs, env_name)
    eval_script_list = make_eval_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )
    eval_script_list_alltests = make_eval_script_list_alltests(
        instance, specs, env_name, repo_directory, base_commit
    )
    eval_script_list_coverage = make_eval_script_list_coverage(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )
    eval_script_list_function = make_eval_script_list_function(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )
    if platform.machine() in {"aarch64", "arm64"}:
        # use arm64 unless explicitly specified
        arch = "arm64" if instance_id not in USE_X86 else "x86_64"
    else:
        arch = "x86_64"

    return TestSpec(
        instance_id=instance_id,
        base_commit=base_commit,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        eval_script_list_alltests=eval_script_list_alltests,
        eval_script_list_coverage=eval_script_list_coverage,
        eval_script_list_function=eval_script_list_function,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        efficiency_test=efficiency_test,
        is_eval=is_eval,
        patch = patch,
        test_patch=test_patch,
    )
