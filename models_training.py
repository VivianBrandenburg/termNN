# -*- coding: utf-8 -*-

from termNN_tools.models import choose_model, get_variables
from termNN_tools.encoding import choose_input
from termNN_tools.toolbox import all_models, seq_length, check_and_make_dir


ks = 10

for RNA_type in ['terminators', 'tRNAs']:
    
    # set input path and output path
    PATH_trainig_data = 'data/' + RNA_type + '/training/'
    PATH_pretrainig_data = 'data/' + RNA_type + '/pretrainig/'
    PATH_model_out =  'models/' + RNA_type 
           
    for  m in all_models: 
        
        # set model handles
        model_name = m['name']
        make_model = choose_model(model_name)
        
        for k in range(ks):                        
            # =================================================================
            # pretraining 
            # =================================================================
            if m['pretrained']:
                # read in pretraining data          
                get_data = choose_input(model_name)
                x_train, y_train = get_data(PATH_pretrainig_data+'train.csv')
                x_test, y_test = get_data(PATH_pretrainig_data+'test.csv')
                
                # build model
                model = make_model(seq_length[RNA_type])
                
                # train model
                early_stop, batch_size = get_variables(model_name, RNA_type, pretraining=True)
                his = model.fit(x_train, y_train, batch_size=batch_size, epochs=100,
                                callbacks = [early_stop], shuffle=True, 
                                validation_data = (x_test, y_test))
            
            # =================================================================
            # main training 
            # =================================================================
            else:
                # build model
                model = make_model(seq_length[RNA_type])
            
            # read in data
            get_data  = choose_input(model_name)
            train_path = PATH_trainig_data +'k'+str(k)+'/'
            x_train, y_train = get_data(train_path + 'train.csv')
            x_test, y_test = get_data(train_path + 'test.csv')
            
            # train model
            early_stop, batch_size = get_variables(model_name, RNA_type)
            his = model.fit(x_train,y_train, batch_size=batch_size, epochs=100,
                            callbacks = [early_stop], shuffle=True,
                            validation_data = (x_test, y_test))
        
            # =================================================================
            # save model 
            # =================================================================
            mydir = f'{PATH_model_out}/{model_name}/'
            check_and_make_dir(mydir)
            model.save(f'{mydir}{model_name}_k{str(k)}.h5')
            
      
            