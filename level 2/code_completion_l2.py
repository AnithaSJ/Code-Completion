import tflearn
import numpy

class Code_Completion:

    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]
    
    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}
    
    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
        return vector
    
    def one_hot_pair(self, string):
        vector = [0] * len(self.string_to_number_pair)
        vector[self.string_to_number_pair[string]] = 1
        return vector
    
    def prepare_data(self, token_lists):
        # encode tokens into one-hot vectors
        all_token_strings = set()    #removes duplicates
        allpair_token_strings = set()    #removes duplicates
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                all_token_strings.add(self.token_to_string(token))
                if idx > 1:
                    two_previous_token_strings = self.token_to_string(token_list[idx - 1]) + self.token_to_string(token_list[idx - 2])
                    allpair_token_strings.add(two_previous_token_strings)    

        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings)))
        self.string_to_number = dict()
        self.number_to_string = dict() 
        max_number = 0
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number    # sample -> {'Boolean-@@-false': 0, 'Boolean-@@-true': 1}
            self.number_to_string[max_number] = token_string    # sample -> {0: 'Boolean-@@-false', 1: 'Boolean-@@-true'}
            max_number += 1
        
        allpair_token_strings = list(allpair_token_strings)
        allpair_token_strings.sort()
        print("Unique token pairs: " + str(len(allpair_token_strings)))
        self.string_to_number_pair = dict()
        self.number_to_string_pair = dict() 
        max_number = 0
        for token_string in allpair_token_strings:
            self.string_to_number_pair[token_string] = max_number    # sample -> {'Boolean-@@-false': 0, 'Boolean-@@-true': 1}
            self.number_to_string_pair[max_number] = token_string    # sample -> {0: 'Boolean-@@-false', 1: 'Boolean-@@-true'}
            max_number += 1

        # prepare x,y pairs
        xs = []
        ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                if idx > 1: 
                    token_string = self.token_to_string(token)
                    previous_token_string = self.token_to_string(token_list[idx - 1]) + self.token_to_string(token_list[idx - 2])
                    xs.append(self.one_hot_pair(previous_token_string))
                    ys.append(self.one_hot(token_string))
                    
        print("x,y pairs: " + str(len(xs)))       
        xs = numpy.array(xs)
        ys = numpy.array(ys)
        xs = xs.reshape([-1,1,len(self.string_to_number_pair)]) 
        return (xs, ys)
       
    def create_network(self):
        self.net = tflearn.input_data(shape=[None,1, len(self.string_to_number_pair)]) # length of unique tokens ( _,86) 
        self.net = tflearn.lstm(self.net, 128, return_seq = True)
        self.net = tflearn.dropout(self.net, 0.5)
#        self.net = tflearn.fully_connected(self.net, 32) # (32, 86) (layer_out,  layer_in)
        self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='softmax') # (86, 32)
        self.net = tflearn.regression(self.net, optimizer= 'adam', loss = 'mean_square', learning_rate=0.001)
        self.model = tflearn.DNN(self.net)  # Our neural network
 
    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file) # load the trained network/model
    
    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists) # generate (x,y) pairs. x is the previous token of y.
        self.create_network()
        self.model.fit(xs, ys, n_epoch=1, batch_size=1024, show_metric=True, shuffle=True, validation_set=0.1) #MA shuffle, validation set
        self.model.save(model_file)   # save the trained network/model
        
    def query(self, prefix, suffix, holes): 
        previous_token_string = self.token_to_string(prefix[-1]) + self.token_to_string(prefix[-2])# prefix[-1] fetches the last element
        x = self.one_hot_pair(previous_token_string)
        x = numpy.array(x)
        x = x.reshape([-1,1,len(self.string_to_number_pair)])
        y = self.model.predict(x)
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist() 
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token1 = self.string_to_token(best_string)
        best_token = [best_token1]
        #***********************MA*****************
        if holes == 2:
            previous_token_string = self.token_to_string(best_token1) + self.token_to_string(prefix[-1])
            x = self.one_hot_pair(previous_token_string)
            x = numpy.array(x)
            x = x.reshape([-1,1,len(self.string_to_number_pair)])
            y = self.model.predict(x)
            predicted_seq = y[0]
            if type(predicted_seq) is numpy.ndarray:
                predicted_seq = predicted_seq.tolist() 
            best_number = predicted_seq.index(max(predicted_seq))
            best_string = self.number_to_string[best_number]
            best_token2 = self.string_to_token(best_string)
            best_token.append(best_token2)
        
        return best_token
    
