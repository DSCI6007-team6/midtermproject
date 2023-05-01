FROM public.ecr.aws/docker/library/python:3.9-slim-buster

RUN apt-get update && apt-get install git -y && apt-get install apt-utils -y && apt-get install libgomp1 -y

WORKDIR /app

COPY  . .

RUN pip3 install -r requirements.txt

RUN mkdir uploads

CMD ["python","app.py"]
