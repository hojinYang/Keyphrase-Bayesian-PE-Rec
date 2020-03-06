#python gethetreclastfm.py
#python gettags.py --dir data/lastfm --cv_id 1 --test
#python gettags.py --dir data/lastfm --cv_id 2 --test
#python gettags.py --dir data/lastfm --cv_id 3 --test
#python gettags.py --dir data/lastfm --cv_id 4 --test
#python gettags.py --dir data/lastfm --cv_id 5 --test

#python main.py --dpath data/lastfm --test --spath table/lastfm
#python main.py --dpath data/lastfm --test --spath table/lastfm --sname main-type2.csv --fold 2 
#python main.py --dpath data/lastfm --test --spath table/lastfm --sname main-type2.csv --fold 1 
#python main.py --dpath data/lastfm --test --spath table/lastfm --sname main-type2.csv --fold 3 
#python main.py --dpath data/lastfm --test --spath table/lastfm --sname main-type2.csv --fold 4 
#python main.py --dpath data/lastfm --test --spath table/lastfm --sname main-type2.csv --fold 5 

python main.py --dpath data/lastfm --test --spath table/lastfm --sname main-all-prec_item.csv --fold 1 --sim_type all --query UCB --prec_item 0.005 --prec_tag 1

