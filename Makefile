error:
	@echo "Empty target is not allowed. Choose one of the targets in the Makefile."
	@exit 2

venv:
	python3 -m venv ./venv
	ln -s venv/bin/activate activate

install_package:
	. ./venv/bin/activate; \
	pip3 install -e .

install_test:
	. ./venv/bin/activate; \
	pip3 install pytest flake8

install_tools:
	. ./venv/bin/activate; \
	pip3 install seaborn scikit-image imageio

install: venv install_package install_test
 
dev: venv install_package install_test install_tools

flake8:
	flake8 --ignore=E501,F401,E128,E402,E731,F821 tests

clean:
	rm -rf `find experiment_interface -name '*.pyc'`
	rm -rf `find experiment_interface -name __pycache__`
	rm -rf `find tests -name '*.pyc'`
	rm -rf `find tests -name __pycache__`

clean_all: clean
	rm -rf activate
	rm -rf venv/
	rm -rf data/
	rm -rf `find . -name '*.log'`
	rm -rf *.egg-info

test:
	pytest tests -s

ci:
	pytest tests

dry_sync: clean
	rsync -anv ${PWD} ${REMOTE_IP}:~/projects/ --exclude='venv/' --exclude='activate' --exclude='experiment_interface.egg-info' --delete

sync: clean
	rsync -azP ${PWD} ${REMOTE_IP}:~/projects/ --exclude='venv/' --exclude='activate' --exclude='experiment_interface.egg-info' --delete


.PHONY: venv install_package install_test install_tools intall dev flake8 clean clean_all test ci dry_sync sync
