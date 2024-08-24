# ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "apple" --output-path "teatime/apple"
# ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "bag of cookies" --output-path "teatime/bag of cookies"
# ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "coffee mug" --output-path "teatime/coffee mug"
# ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "cookies on a plate" --output-path "teatime/cookies on a plate"
# ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "paper napkin" --output-path "teatime/paper napkin"
# ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "plate" --output-path "teatime/plate"
# ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "sheep" --output-path "teatime/sheep"
# ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "spoon handle" --output-path "teatime/spoon handle"
# ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "stuffed bear" --output-path "teatime/stuffed bear"
# ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "tea in a glass" --output-path "teatime/tea in a glass"
CUDA_VISIBLE_DEVICES=2 ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "which is red fruit" --output-path "lerf_masks/teatime_reasoning/apple"
CUDA_VISIBLE_DEVICES=2 ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "which is the brown bag on the side of the plate" --output-path "lerf_masks/teatime_reasoning/bag of cookies"
CUDA_VISIBLE_DEVICES=2 ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "which cup is used for coffee" --output-path "lerf_masks/teatime_reasoning/coffee mug"
CUDA_VISIBLE_DEVICES=2 ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "which are the cookies" --output-path "lerf_masks/teatime_reasoning/cookies on a plate"
CUDA_VISIBLE_DEVICES=2 ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "what can be used to wipe hands" --output-path "lerf_masks/teatime_reasoning/paper napkin"
CUDA_VISIBLE_DEVICES=2 ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "what can be used to hold cookies" --output-path "lerf_masks/teatime_reasoning/plate"
CUDA_VISIBLE_DEVICES=2 ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "which is a cute white doll" --output-path "lerf_masks/teatime_reasoning/sheep"
CUDA_VISIBLE_DEVICES=2 ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "which is spoon handle" --output-path "lerf_masks/teatime_reasoning/spoon handle"
CUDA_VISIBLE_DEVICES=2 ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "which is the brown bear doll" --output-path "lerf_masks/teatime_reasoning/stuffed bear"
CUDA_VISIBLE_DEVICES=2 ns-render dataset --load-config outputs/teatime/lerf/2024-08-22_111901/config.yml --rendered-output-names mask_map_0 --colormap-options.colormap turbo --colormap-options.colormap-min -1 --colormap-options.normalize True --text_prompt "which is the drink in the transparent glass" --output-path "lerf_masks/teatime_reasoning/tea in a glass"

