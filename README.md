# Preeminence

A turn-based strategy game for AIs to play against each other.

Get started using the agent [tutorial](https://douglasorr.github.io/Preeminence/tutorial.html), or peruse our [documentation](https://douglasorr.github.io/Preeminence/).


## Publishing

    # 1. Check everything works
    ./run build
    ./run check
    ./run docs --examples

    # 2. Publish Docker image
    VERSION=0.1
    docker login
    docker tag preeminence douglasorr/preeminence:$VERSION
    docker push douglasorr/preeminence:$VERSION
    docker tag preeminence douglasorr/preeminence:latest
    docker push douglasorr/preeminence:latest

    # 3. Publish docs
    cd docs && git add . && git commit -m "Update docs"
    git push && cd ..

    # 4. Make sure git knows
    git push origin HEAD:refs/tags/$VERSION

    # 5. Check everything worked
    # https://cloud.docker.com/repository/docker/douglasorr/preeminence/tags
    # https://douglasorr.github.io/Preeminence
