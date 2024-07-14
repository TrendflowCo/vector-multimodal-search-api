class FilterBuilder:
    @staticmethod
    def build_filters(params):
        filters = []
        if params.get('min_price') is not None:
            filters.append({"path": ["price"], "operator": "GreaterThanEqual", "valueNumber": params['min_price']})
        if params.get('max_price') is not None:
            filters.append({"path": ["price"], "operator": "LessThanEqual", "valueNumber": params['max_price']})
        if params.get('tags'):
            tags_list = [x.lower() for x in params['tags'].replace("'", "").split(',')]
            filters.append({"path": ["tags"], "operator": "ContainsAny", "valueText": tags_list})
        if params.get('brands'):
            brands_list = [x.lower() for x in params['brands'].replace("'", "").split(',')]
            filters.append({"path": ["brand"], "operator": "ContainsAny", "valueText": brands_list})
        if params.get('category'):
            filters.append({"path": ["category"], "operator": "Equal", "valueText": params['category']})
        if params.get('on_sale'):
            filters.append({"path": ["onSale"], "operator": "Equal", "valueBoolean": params['on_sale']})
        if params.get('list_ids'):
            list_ids = params['list_ids'].replace("'", "").split(',')
            filters.append({"path": ["id"], "operator": "ContainsAny", "valueText": list_ids})
        if params.get('country'):
            filters.append({"path": ["country"], "operator": "Equal", "valueText": params['country']})
        if params.get('currency'):
            filters.append({"path": ["currency"], "operator": "Equal", "valueText": params['currency']})
        return filters
