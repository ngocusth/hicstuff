# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
FROM python:alpine




LABEL Name=hicstuff Version=1.4.5

# Install dependencies
COPY * ./ /app/
WORKDIR /app
RUN pip3 install -Ur requirements.txt

# Using pip:
RUN pip3 install .
#CMD ["python3", "-m", "hicstuff.main"]
ENTRYPOINT [ "hicstuff" ]