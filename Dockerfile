FROM python:3.10-slim-buster

RUN pip install explainerdashboard
RUN pip install dill

COPY dashboard.py ./
COPY app.py ./
COPY model.py ./
COPY eda.py ./

RUN python eda.py
RUN python model.py
RUN python dashboard.py

EXPOSE 9050
CMD ["python", "./app.py"]
