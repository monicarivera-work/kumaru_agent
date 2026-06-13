from setuptools import setup, find_packages

setup(
    name="kumaru-agent",
    version="0.1.0",
    description="An enterprise LLM agent built from scratch for learning purposes",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "typing_extensions>=4.9.0",
    ],
)
