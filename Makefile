run:
	streamlit run app/streamlit_app.py
eval:
	python src/eval.py data/*.pdf
tune:
	python src/tune_tau.py data/*.pdf
test:
	pytest -q
