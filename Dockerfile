FROM opencv4.5.5-cuda:11.3.0-cudnn8-devel-ubuntu20.04

RUN mv /usr/local/lib/python3.8/site-packages/cv2/ /usr/local/lib/python3.8/dist-packages/

COPY . /home/yolov4_tracking_apps/

RUN cd /home/yolov4_tracking_apps/ && pip3 install -r requirements.txt --no-deps


#RUN sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
#RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
#RUN apt-get update && apt-get install -y postgresql-14 postgresql-client-14 postgresql-contrib-14
#USER postgres
#RUN    /etc/init.d/postgresql start &&\
#    psql --command "CREATE USER docker WITH SUPERUSER PASSWORD 'docker';" &&\
#    createdb -O face_recognition docker
#RUN echo "host all  all    0.0.0.0/0  md5" >> /etc/postgresql/14/main/pg_hba.conf
#RUN echo "listen_addresses='*'" >> /etc/postgresql/14/main/postgresql.conf
#EXPOSE 5432
#VOLUME  ["/etc/postgresql", "/var/log/postgresql", "/var/lib/postgresql"]
#CMD ["/usr/lib/postgresql/14/bin/postgres", "-D", "/var/lib/postgresql/14/main", "-c", "config_file=/etc/postgresql/14/main/postgresql.conf"]


WORKDIR /home/yolov4_tracking_apps

#CMD python3 pgsql_script.py

#ENTRYPOINT python3 mot_server.py
#ENTRYPOINT ["python3", "mot_server.py"]