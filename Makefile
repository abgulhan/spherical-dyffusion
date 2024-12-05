.PHONY: deps_table_update modified_only_fixup quality style

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := scripts src utils

modified_only_fixup:
	$(eval modified_py_files := $(shell python3 utils/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		black $(modified_py_files); \
		ruff check $(modified_py_files); \
	else \
		echo "No library .py files were modified"; \
	fi

# Update src/dependency_versions_table.py

deps_table_update:
	@python3 setup.py "deps_table_update"

deps_table_check_updated:
	@md5sum src/dependency_versions_table.py > md5sum.saved
	@python3 setup.py deps_table_update
	@md5sum -c --quiet md5sum.saved || (printf "\nError: the version dependency table is outdated.\nPlease run 'make fixup' or 'make style' and commit the changes.\n\n" && exit 1)
	@rm md5sum.saved

# autogenerating code

autogenerate_code: deps_table_update

# this target runs checks on all files

quality:
	python3 -m black --check $(check_dirs)
	python3 -m ruff check $(check_dirs)
# 	doc-builder style src docs/source --max_len 119 --check_only --path_to_docs docs/source


# this target runs checks on all files and potentially modifies some of them

style:
	python3 -m black $(check_dirs)
	python3 -m ruff check $(check_dirs) --fix
	${MAKE} autogenerate_code
