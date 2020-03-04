#python getmovielens.py
#python gettags.py --dir data/movielens --cv_id 1 --test
#python gettags.py --dir data/movielens --cv_id 2 --test
#python gettags.py --dir data/movielens --cv_id 3 --test
#python gettags.py --dir data/movielens --cv_id 4 --test
#python gettags.py --dir data/movielens --cv_id 5 --test

#python main.py --dpath data/movielens --test --spath table/movielens --sname main-type2.csv --fold 2 

python main.py --dpath data/movielens --test --spath table/movielens --sname main-type2.csv --fold 3 
python main.py --dpath data/movielens --test --spath table/movielens --sname main-type2.csv --fold 4 
python main.py --dpath data/movielens --test --spath table/movielens --sname main-type2.csv --fold 5 

python main.py --dpath data/movielens --test --spath table/movielens --sname main-type2.csv --fold 1 