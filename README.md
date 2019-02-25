# Preeminence

A turn-based strategy game for AIs to play against each other.

Get started using the agent [tutorial](https://douglasorr.github.io/Preeminence/tutorial.html), or peruse our [documentation](https://douglasorr.github.io/Preeminence/).


## Publishing

    # Publish Docker image
    VERSION=0.1
    ./run build
    ./run check
    docker login
    docker tag preeminence douglasorr/preeminence:$VERSION
    docker push douglasorr/preeminence:$VERSION
    docker tag preeminence douglasorr/preeminence:latest
    docker push douglasorr/preeminence:latest

    # Publish docs
    ./run docs --examples
    cd docs && git add . && git commit -m "Update docs"
    git push
