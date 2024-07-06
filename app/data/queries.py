combined_query = """
SELECT *
FROM `dokuso.production.combined`
"""

combined_tags_query = """
SELECT *
FROM `dokuso.production.combined_tags`
where value is not null
"""