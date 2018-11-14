PACKAGE_NAME = 'experiment_interface'

error:
	@echo "Empty target is not allowed. Choose one of the targets in the Makefile."
	@exit 2

venv:
	python3 -m venv ./venv
	ln -s venv/bin/activate activate
	. ./venv/bin/activate; \
	pip3 install -U pip setuptools wheel

install_package:
	. ./venv/bin/activate; \
	pip3 install -e .

install_test:
	. ./venv/bin/activate; \
	pip3 install -U pytest flake8

install_tools:
	. ./venv/bin/activate; \
	pip3 install -U seaborn scikit-image imageio

install: venv install_package install_test
 
dev: venv install_package install_test install_tools

test:
	pytest tests -s

ci:
	pytest tests

flake8:
	flake8 --ignore=E501,F401,E128,E402,E731,F821 experiment_interface
	flake8 --ignore=E501,F401,E128,E402,E731,F821 tests

clean:
	rm -rf `find $(PACKAGE_NAME) -name '*.pyc'`
	rm -rf `find $(PACKAGE_NAME) -name __pycache__`
	rm -rf `find tests -name '*.pyc'`
	rm -rf `find tests -name __pycache__`

clean_all: clean
	rm -rf *.egg-info
	rm -rf venv/
	rm -rf activate
	rm -rf `find . -name '*.log'`

.PHONY: venv install_package install_test install_tools intall dev flake8 clean clean_all test ci dry_sync sync

# Tools for setting up / syncing remote ubuntu server.

setup:
	scp ${HOME}/projects/scripts/_dircolors ${REMOTE_IP}:~/.dircolors
	scp ${HOME}/projects/scripts/_bash_custom ${REMOTE_IP}:~/.bash_custom
	ssh ${REMOTE_IP} "mkdir -p projects"
	ssh ${REMOTE_IP} "sudo apt-get install python3-venv"

dry_sync: clean
	rsync -anv ${PWD} ${REMOTE_IP}:~/projects/ --exclude-from='~/projects/rsync_exclude.txt'

sync: clean
	rsync -azP ${PWD} ${REMOTE_IP}:~/projects/ --exclude-from='~/projects/rsync_exclude.txt'


