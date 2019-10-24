FROM ufoym/deepo:all-py36-jupyter-cpu

RUN pip install numpy scipy pandas scikit-learn streamlit

WORKDIR /data

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD ["streamlit", "run", "notebook.py"]
