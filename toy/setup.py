from setuptools import setup, find_packages


def main():
    setup(
        name="nvtc_toy",
        author="Runsen Feng",
        author_email="fengruns@mail.ustc.edu.cn",
        python_requires=">=3.10",
        packages=find_packages(),
    )


if __name__ == "__main__":
    main()
