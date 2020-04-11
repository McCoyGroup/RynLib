
# If not building on top of entos, build DockerfileCore as rynlibcore
# otherwise use DockerfileEntos and build from that

# FROM rynlibcore
FROM entos

#
#   RYNLIB:
#       Defines how to load RynLib inside the Dockerized environment
#       This chunk can easily be a publicly available thing
#

#RyAN AND MARK HAFVE FUN TOGETHER

ADD . /home/RynLib

RUN \
  git clone https://github.com/McCoyGroup/Peeves /home/Peeves

RUN python3 /home/RynLib/CLI.py config install-mpi --version=3.2 --imp=mpich

RUN python3 /home/RynLib/CLI.py config build-libs

WORKDIR /home

ENTRYPOINT ["python3", "/home/RynLib/CLI.py"]