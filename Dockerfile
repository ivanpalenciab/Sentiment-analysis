FROM python:3.12

RUN pip install --no-cache-dir jupyter pandas scikit-learn matplotlib keras tensorflow

COPY . .

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root" ]