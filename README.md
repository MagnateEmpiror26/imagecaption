# Tushtoken

pane cell rine line of code inoti "tokenizer.fit_text". after that cell write code iyo top save the tokenizer object. the file will be put in the current working directory. move it to the folder rine app.py yako.

# Saving the tokenizer object for future use
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


there is a cell rine line of code 'model = Dense...'
then yonzi fe = Model(input_shape=....

after this cell 

write fe.save('feature_extractor.h5')

then it will create a model mu current working directory move it ku folder rine app.py

the last one ndatokupa ukaisa

by now everything must be set.

isa tensorflow mu requirements.txt
