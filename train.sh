CUDA_VISIBLE_DEVICES=1 ns-train lerf --data data/gsgrouping/figurines
CUDA_VISIBLE_DEVICES=2 ns-train lerf --data data/gsgrouping/ramen
CUDA_VISIBLE_DEVICES=3 ns-train lerf --data data/gsgrouping/teatime

CUDA_VISIBLE_DEVICES=4 ns-train lerf --data data/ovs3d/bed
CUDA_VISIBLE_DEVICES=1 ns-train lerf --data data/ovs3d/bench
CUDA_VISIBLE_DEVICES=1 ns-train lerf --data data/ovs3d/lawn
CUDA_VISIBLE_DEVICES=1 ns-train lerf --data data/ovs3d/room
CUDA_VISIBLE_DEVICES=2 ns-train lerf --data data/ovs3d/sofa

CUDA_VISIBLE_DEVICES=1 ns-train lerf_ovs3d --data data/ovs3d/ns_room

CUDA_VISIBLE_DEVICES=4 ns-train lerf --data data/messy_rooms/large_corridor_25