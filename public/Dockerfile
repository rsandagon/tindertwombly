FROM nginx:stable-alpine
COPY default.conf /etc/nginx/conf.d
COPY index.html /usr/share/nginx/html/index.html
COPY main.js /usr/share/nginx/html/main.js
COPY main.css /usr/share/nginx/html/main.css
COPY ./icons /usr/share/nginx/html
EXPOSE 8080
EXPOSE 80
CMD ["/bin/sh", "-c", "exec nginx -g 'daemon off;';"]
WORKDIR /usr/share/nginx/html