#python getmovielens.py
#python gettags.py --dir data/movielens --cv_id 1 --test
#python gettags.py --dir data/movielens --cv_id 2 --test
#python gettags.py --dir data/movielens --cv_id 3 --test
#python gettags.py --dir data/movielens --cv_id 4 --test
#python gettags.py --dir data/movielens --cv_id 5 --test

#python main.py --dpath data/movielens --test --spath table/movielens --sname main-type2.csv --fold 2 
#python main.py --dpath data/movielens --test --spath table/movielens --sname main-type2.csv --fold 3 
#python main.py --dpath data/movielens --test --spath table/movielens --sname main-type2.csv --fold 4 
#python main.py --dpath data/movielens --test --spath table/movielens --sname main-type2.csv --fold 5 

#python main.py --dpath data/movielens --test --spath table/movielens --sname main-type2.csv --fold 1 
#python main.py --dpath data/movielens --test --spath table/movielens --prec_item 0.001 --prec_tag 100000 --sname main-type2-prec-tag-test.csv --fold 1 --query UCB
#python main.py --dpath data/movielens --test --spath table/movielens --prec_item 0.001 --prec_tag 0.001 --sname main-type2-prec-tag-test.csv --fold 1 --query UCB
#python main.py --dpath data/movielens --test --spath table/movielens --prec_item 0.001 --prec_tag 0.1 --sname main-type2-prec-tag-test.csv --fold 1 --query UCB
#python main.py --dpath data/movielens --test --spath table/movielens --prec_item 0.001 --prec_tag 1000000000 --sname main-type2-prec-tag-test.csv --fold 1 --query UCB


#python main.py --dpath data/movielens --test --spath table/movielens --sname main-type1.csv --fold 1 --sim_type subset
#python main.py --dpath data/movielens --test --spath table/movielens --sname main-type1.csv --fold 2 --sim_type subset
#python main.py --dpath data/movielens --test --spath table/movielens --sname main-type1.csv --fold 3 --sim_type subset
#python main.py --dpath data/movielens --test --spath table/movielens --sname main-type1.csv --fold 4 --sim_type subset
#python main.py --dpath data/movielens --test --spath table/movielens --sname main-type1.csv --fold 5 --sim_type subset
python main.py --dpath data/movielens --test --spath table/movielens --sname main-type1-fix3.csv --fold 1 --sim_type all --query UCB Mean --prec_tag 0.1
python main.py --dpath data/movielens --test --spath table/movielens --sname main-type1-fix3.csv --fold 2 --sim_type all --query UCB Mean