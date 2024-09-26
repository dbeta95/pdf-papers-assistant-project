docker_run_elasticsearch:
	docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
	-v elasticsearch-data:/usr/share/elasticsearch/data \
    elasticsearch:8.4.3
