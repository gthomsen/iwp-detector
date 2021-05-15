#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="iwp",
    version="0.0.1",
    description="Internal Wave Packet Tools",
    url="https://github.com/gthomsen/iwp",
    author="Greg Thomsen",
    packages=find_packages( include=["iwp"] )
    )
