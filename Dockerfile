# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
FROM python:3.7




LABEL Name=hicstuff Version=1.4.7

# Install python dependencies
COPY * ./ /app/
WORKDIR /app
RUN pip3 install -Ur requirements.txt

# System packages 
RUN apt-get update && apt-get install -y curl

# Install miniconda to /miniconda
RUN curl -LO https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Get 3rd party packages from conda
RUN conda install -c bioconda -y \
    bowtie2 \
    minimap2 \
    samtools

# Using pip:
RUN pip3 install .
#CMD ["python3", "-m", "hicstuff.main"]
ENTRYPOINT [ "hicstuff" ]