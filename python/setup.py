#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="iwp",
    version="0.0.1",
    description="Internal Wave Packet Tools",
    url="https://github.com/gthomsen/iwp-detector",
    author="Greg Thomsen",
    packages=find_packages( include=["iwp"] ),
    scripts=[
        "iwp/scripts/iwp_compute_statistics.py",
        "iwp/scripts/iwp_create_labeling_data.py",
        "iwp/scripts/iwp_export_pptx.py",
        "iwp/scripts/iwp_merge_labels.py",
        "iwp/scripts/iwp_postprocess_netcdf.py",
        "iwp/scripts/scalabel_extract_iwp_labels.py",
        "iwp/scripts/scalabel_generate_playlist.py"
    ]
    )
