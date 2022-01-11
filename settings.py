import json

# settings_json = json.dumps(
#     [
#         {
#             'type': 'title',
#             'title': 'Classification Method'
#         },
#         {
#             'type': 'options',
#             'title': 'Method',
#             'desc': 'You can choose either standalone model or voting committee',
#             'section': 'options',
#             'key': 'boolconfiguration',
#             'options': ['classifier', 'committee']
#         },
#         {
#             'type': 'title',
#             ',title': 'Standalone Classification'
#         },
#     ]
# )

settings_json = json.dumps(
    [
        {
            'type': 'title',
            'title': 'Classification Method'
        },
        {
            'type': 'options',
            'title': 'Method',
            'desc': 'Either single classifier or committee',
            'section': 'configuration',
            'key': 'method',
            'options': ['classifier', 'committee']
        },
        {
            'type': 'title',
            'title': "Classifier"
        },
        {
            'type': 'options',
            'title': 'Model',
            'desc': 'Model used for classification',
            'section': 'configuration',
            'key': 'model',
            'options':
            [
                'CNN',
                'CRNN',
                'CRNN2',
                'CRNN3',
                'CRNN4',
                'CRNN5',
                'CRNN6',
                'CRNN7',
                'CRNN8',
                'PRCNN'
            ]
        },
        {
            'type': 'title',
            'title': "Voting Committee"
        },
        {
            'type': 'options',
            'title': 'Voting Method',
            'section': 'configuration',
            'key': 'voting',
            'options': ['naive', 'kApproval', 'majority']
        },
        {
            'type': 'bool',
            'title': 'CNN',
            'section': 'configuration',
            'key': 'cnn'
        },
        {
            'type': 'bool',
            'title': 'CRNN1',
            'section': 'configuration',
            'key': 'crnn'
        },
        {
            'type': 'bool',
            'title': 'CRNN2',
            'section': 'configuration',
            'key': 'crnn2'
        },
        {
            'type': 'bool',
            'title': 'CRNN3',
            'section': 'configuration',
            'key': 'crnn3'
        },
        {
            'type': 'bool',
            'title': 'CRNN4',
            'section': 'configuration',
            'key': 'crnn4'
        },
        {
            'type': 'bool',
            'title': 'CRNN5',
            'section': 'configuration',
            'key': 'crnn5'
        },
        {
            'type': 'bool',
            'title': 'CRNN6',
            'section': 'configuration',
            'key': 'crnn6'
        },
        {
            'type': 'bool',
            'title': 'CRNN7',
            'section': 'configuration',
            'key': 'crnn7'
        },
        {
            'type': 'bool',
            'title': 'CRNN8',
            'section': 'configuration',
            'key': 'crnn8'
        },
        {
            'type': 'bool',
            'title': 'PRCNN',
            'section': 'configuration',
            'key': 'prcnn'
        }

    ]
)
