"""Activation names the MLPs accept, kept free of any backend import.

Config entrypoints validate a swept value before submitting a job, and they run under whatever
interpreter launches the sweep -- which need not have jax installed. Importing the backend just to
read a list of names would make that validation the thing that fails.
"""
ACTIVATION_NAMES = ("tanh", "softplus", "relu", "elu", "silu", "gelu")
