
FROM entos

#
#   RYNLIB:
#       Defines how to load RynLib inside the Dockerized environment
#       This chunk can easily be a publicly available thing
#

#RyAN AND MARK HAFVE FUN TOGETHER

RUN \
  apt-get -y install python3.7-dev python3-pip &&\
  pip3 install numpy &&\
  pip3 install wget

RUN \
  git clone https://github.com/McCoyGroup/RynLib /home/RynLib &&\
  git clone https://github.com/McCoyGroup/Peeves /home/Peeves

#RUN \
#  python3 /home/Ryna/Documents/RynLib/build.py

WORKDIR /home

ENTRYPOINT ["python3", "/home/RynLib/CLI.py"]