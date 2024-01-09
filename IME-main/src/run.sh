#IME-main
python3 IME-main.py \
--ProjectedModel='Projection_center' \
--output_file='../results/calculated_coordinates_method1.csv' \

python3 IME-main.py \
--ProjectedModel='Center_projection' \
--output_file='../results/calculated_coordinates_method2.csv' \

python3 plot_data1.py \
--filename_method1='../results/calculated_coordinates_method1.csv' \
--filename_method2='../results/calculated_coordinates_method2.csv' \