FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y python3-numpy python3-scipy python3-pip build-essential git
RUN echo "[global] \n\
timeout = 60 \n\
index = https://pypi.tuna.tsinghua.edu.cn/ \n\
index-url = https://pypi.tuna.tsinghua.edu.cn/simple/ \n\
trusted-host = pypi.tuna.tsinghua.edu.cn \n\
[list] \n\
format = columns" > /etc/pip.conf && cat /etc/pip.conf 
RUN pip3 install -U pip

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip3 install -rrequirements.txt

ENTRYPOINT ["python3", "run_algorithm.py"]
