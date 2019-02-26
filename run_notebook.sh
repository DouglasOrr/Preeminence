NAME=${USER}-preem
PORT=8888

docker pull douglasorr/preeminence
docker rm -f ${NAME} &> /dev/null || true
docker run -d --name ${NAME} -p ${PORT}:${PORT} -e PYTHONPATH=/preem -v `pwd`:/preem -w /preem douglasorr/preeminence jupyter notebook --allow-root --port ${PORT} --ip 0.0.0.0 > /dev/null

sleep 3

TOKEN=$(docker logs ${NAME} 2>&1 | grep -oE '\?token=.+$' | head -1)
echo "See:"
echo "    http://localhost:${PORT}/notebooks/Tutorial.ipynb${TOKEN}"
echo
echo "When you want to stop your server:"
echo "    docker rm -f ${NAME}"
