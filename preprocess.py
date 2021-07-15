import json
import os



START_TOKEN = '<title_start>'
END_TOKEN = '<title_end/>'
# DATA_DIR = 'data/json/'
# PROJECTS = ['cassandra', 'elasticsearch', 'gradle', 'hadoop-common', 'hibernate-orm', 'intellij-community', 'libgdx', 'liferay-portal', 'presto', 'spring-framework', 'wildfly']
DATA_DIR = 'data/other/'
PROJECTS = ['csn']

def files_to_data(DIR, FILE):
    """
    DIR is the base directory containing all the json files
    FILES is the filename of the json file you want to get the data from
    """
    data = []
    with open(os.path.join(DIR, FILE), 'r') as r:
        project = json.load(r)
        for method in project:
            assert type(method['filename']) is str
            assert type(method['name']) is list
            assert type(method['tokens']) == list
            method_name = [START_TOKEN] + method['name'] + [END_TOKEN] #add start and end of sequence token to method name
            method_body = [x.lower() for x in method['tokens'] if (x != '<id>' and x != '</id>')] #lowercase and remove <id> tags
            
            #when the method name appears in the body it is represented by a %self% token
            #we replace the %self% token with the actual method name
            while '%self%' in method_body: 
                self_idx = method_body.index('%self%')
                method_body = method_body[:self_idx] + method['name'] + method_body[self_idx+1:]
       
            data.append({'name': method_name[:13], 'body': method_body[:100]})

    return data

for project in PROJECTS:

    print(f'Project: {project}')

    train_file = f'{project}_train_methodnaming.json'
    test_file = f'{project}_test_methodnaming.json'

    train_data = files_to_data(DATA_DIR, train_file)
    test_data = files_to_data(DATA_DIR, test_file)

    print(f'Training examples: {len(train_data)}')
    print(f'Testing examples: {len(test_data)}')
    
    with open(f'data/{project}_train.json', 'w') as w:
        for example in train_data:
            json.dump(example, w)
            w.write('\n')

    with open(f'data/{project}_test.json', 'w') as w:
        for example in test_data:
            json.dump(example, w)
            w.write('\n')
