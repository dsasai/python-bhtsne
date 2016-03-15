
all:
	python setup.py build_ext --inplace

.PHONY: clean
clean:
	rm -f bhtsne_wrapper.cpp
	rm -f bhtsne_wrapper.so

.PHONY: test
test:
	cd test; PYTHONPATH="../" python -m unittest tsne
