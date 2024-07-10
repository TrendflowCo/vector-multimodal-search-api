class FilterBuilder:
    @staticmethod
    def build_filters(params):
        filters = []
        if params.get('min_price') is not None:
            filters.append({"path": ["price"], "operator": "GreaterThanEqual", "valueNumber": params['min_price']})
        if params.get('max_price') is not None:
            filters.append({"path": ["price"], "operator": "LessThanEqual", "valueNumber": params['max_price']})
        if params.get('brands'):
            filters.append({"path": ["brand"], "operator": "Equal", "valueText": params['brands']})
        if params.get('tags'):
            filters.append({"path": ["tags"], "operator": "Equal", "valueText": params['tags']})
        if params.get('category'):
            filters.append({"path": ["category"], "operator": "Equal", "valueText": params['category']})
        if params.get('on_sale'):
            filters.append({"path": ["onSale"], "operator": "Equal", "valueBoolean": params['on_sale']})
        if params.get('list_ids'):
            filters.append({"path": ["id"], "operator": "In", "valueText": params['list_ids'].split(',')})
        if params.get('country'):
            filters.append({"path": ["country"], "operator": "Equal", "valueText": params['country']})
        if params.get('currency'):
            filters.append({"path": ["currency"], "operator": "Equal", "valueText": params['currency']})
        return filters
