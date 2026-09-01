#!/usr/bin/env bash
set -e

sudo ip link set can1 down 2>/dev/null || true
sudo ip link set can3 down 2>/dev/null || true
sudo pkill slcand 2>/dev/null || true

sleep 0.5

test -e /dev/arxcan1
test -e /dev/arxcan3

sudo slcand -o -f -s8 /dev/arxcan1 can1
sudo ip link set can1 up

sudo slcand -o -f -s8 /dev/arxcan3 can3
sudo ip link set can3 up

ip link show can1
ip link show can3