import time
import sagemaker


SIMILAR = 5
UNSIMILAR = 10


_model_name = None
_endpoint_name = None
_session = None
_predictor = None
_messages = dict()
_categories = dict()
_checkpoint = dict()


def deploy_model(model_arn=None, model_name=None, instance_type=None):
    global _model_name, _endpoint_name, _session, _predictor

    if _session is not None:
        print('Previous session is still active, you must shutdown it first. Please use shutdown() function.')
        return

    if model_arn is None and model_name is None:
        raise ValueError('Both model_arn adn model_name is None')

    # create unique model name and endpoint name
    timestamp = str(int(time.time()))
    _model_name = 'dynamic-ai-model-' + timestamp
    _endpoint_name = 'dynamic-ai-endpoint-' + timestamp

    # this line of code requires iam:GetRole permissions
    role = sagemaker.get_execution_role()

    # initialize a SageMaker session
    _session = sagemaker.Session()

    # create a new Model object
    print('Creating model: ' + _model_name)

    if model_arn is not None:
        model = sagemaker.model.ModelPackage(sagemaker_session=_session, name=_model_name, model_package_arn=model_arn, role=role)

    elif model_name is not None:
        account = _session.boto_session.client('sts').get_caller_identity()['Account']
        region = _session.boto_session.region_name

        image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, model_name)
        model = sagemaker.model.Model(sagemaker_session=_session, name=_model_name, image=image, model_data=None, role=role)

    # deploy it to an endpoint
    print('Deploying model to endpoint: ' + _endpoint_name)
    model.deploy(initial_instance_count=1, instance_type=instance_type, endpoint_name=_endpoint_name)

    # construct predictor
    _predictor = sagemaker.predictor.RealTimePredictor(endpoint=_endpoint_name, sagemaker_session=_session,
        serializer=sagemaker.predictor.json_serializer, deserializer=sagemaker.predictor.json_deserializer)

    # we're done
    print('\nSuccess')


def shutdown():
    global _model_name, _endpoint_name, _session, _predictor

    if _predictor is not None:
        _predictor.delete_endpoint(delete_endpoint_config=True)
        _predictor.delete_model()

    _model_name = None
    _endpoint_name = None
    _session = None
    _predictor = None
    _messages.clear()
    _categories.clear()
    _checkpoint.clear()


def get_internal_state():
    return (_model_name, _endpoint_name, _session, _predictor, _messages, _categories, _checkpoint)


def set_internal_state(state):
    global _model_name, _endpoint_name, _session, _predictor, _messages, _categories, _checkpoint
    _model_name, _endpoint_name, _session, _predictor, _messages, _categories, _checkpoint = state


def is_ready():
    response = _predictor.predict({"type":"isReady"})
    return bool(response["ready"])


def reset():
    _predictor.predict({"type":"reset"})

    _messages.clear()
    _categories.clear()
    _checkpoint.clear()


def create_checkpoint():
    response = _predictor.predict({"type":"saveCheckpoint"})
    if response['result'] == 'success':
        _checkpoint['time'] = time.time()
        _checkpoint['messages'] = dict(_messages)
        _checkpoint['categories'] = dict(_categories)
        return True
    return False


def restore_checkpoint():
    if not _checkpoint:
        raise RuntimeError('To respore checkpoint, you should create it first')

    response = _predictor.predict({"type":"restoreCheckpoint"})
    if response['result'] == 'success':
        _messages = dict(_checkpoint['messages'])
        _categories = dict(_checkpoint['categories'])
        return True
    return False


def add_message(message):
    attempt = 0
    max_attempts = 10
    while True:
        try:
            response = _predictor.predict(
                {
                    "type": "addMessage",
                    "message": message,
                    "security_group": "default"
                })

            message_id = response.get('message_id')
            if type(message_id) == str and len(message_id) > 0:
                _messages[message_id] = message
                return message_id;
            return None

        except:
            attempt += 1
            if attempt == max_attempts:
                raise


def _try_add_message(message):
    return _predictor.predict(
        {
            "type": "addMessage",
            "message": message,
            "security_group": "default"
        })


def add_feedback(message_id, relations):
    if message_id not in _messages:
        raise ValueError('Unknown message_id')

    rel = []
    for id, flags in relations:
        rel.append({'id': id, 'flags': flags})

    response = _predictor.predict(
        {
            "type": "addFeedback",
            "message_id": message_id,
            "relations": rel
        })

    return response.get('result') == 'success'


def set_category(message_id, category):
    if message_id not in _messages:
        raise ValueError('Unknown message_id')

    relations = list()
    for id, cat in _categories.items():
        if id == message_id:
            continue
        if cat == category:
            relations.append((id, SIMILAR))
        else:
            relations.append((id, UNSIMILAR))

    _categories[message_id] = category;
    return add_feedback(message_id, relations)


def get_similarity(message_id, accuracy_limit=0.6, block_limit=2):
    response = _predictor.predict(
        {
            "type": "getSimilarity",
            "message_id": message_id,
            "precision_limit": accuracy_limit,
            "block_limit": block_limit
        })
    similarity = response["similarity"]
    return [_translate_similarity(x) for x in similarity if x.get('internalId', '') != message_id]


def _translate_similarity(x):
    message_id = str(x.get('internalId', ''))
    return {
        'message_id': message_id,
        'message_text': _messages.get(message_id, ''),
        'similarity': int(_extract_value(x.get('techReport'), 'bits')),
        'accuracy': round(float(x.get('accuracy')), 1),
        'is_approved': bool(x.get('isApproved')),
        'is_same_text': bool(x.get('theSameText')),
        'has_statistics': bool(x.get('statisticsExist'))
    }


def _extract_value(text, key):
    if text is None or key is None:
        return None
    x = text.split()
    for i in range(0, len(x)-2):
        if x[i] == key and x[i + 1] == '=':
            return x[i + 2]
    return None


def get_tech_report(message_id, accuracy_limit=0.6, block_limit=2):
    response = _predictor.predict(
        {
            "type": "getSimilarity",
            "message_id": message_id,
            "precision_limit": accuracy_limit,
            "block_limit": block_limit
        })
    return response["tech_report"]


def predict_category(message_id, accuracy_limit=0.6):
    if message_id not in _messages:
        raise ValueError('Unknown message_id')

    if message_id in _categories:
        return {
            'category': _categories.get(message_id),
            'accuracy': 1.0,
            'is_approved': True,
        }

    similarity = get_similarity(message_id, accuracy_limit)

    for s in similarity:
        id = s.get('message_id');
        if id not in _categories:
            continue
        if s.get('similarity', 0) < 75:
            continue
        return {
            'category': _categories.get(id),
            'accuracy': s.get('accuracy'),
            'is_approved': s.get('is_approved'),
        }

    return {
        'category': None,
        'accuracy': 0.0,
        'is_approved': False,
    }


def list_messages(category=None):
    if category is None:
        return dict(_messages)
    return {id: _messages.get(id) for id, cat in _categories.items() if cat == category}


def list_categories():
    return dict(_categories)


