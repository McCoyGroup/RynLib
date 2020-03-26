
FROM entos

#
#   RYNLIB:
#       Defines how to load RynLib inside the Dockerized environment
#       This chunk can easily be a publicly available thing
#

#RyAN AND MARK HAFVE FUN TOGETHER

RUN \
  git clone https://github.com/McCoyGroup/RynLib /home/RynLib &&\
  git clone https://github.com/McCoyGroup/Peeves /home/Peeves

WORKDIR /home

ENTRYPOINT ["python3", "/home/RynLib/CLI.py"]