# 导入所需的模块
from py2neo import Graph
import time
import random


# 连接到Neo4j数据库
graph = Graph("bolt://localhost:7687", user="neo4j", password="12345678")


# 查询所有用户
def get_all_users():
    users = graph.run("MATCH (u:User) RETURN u").data()
    return users


# 查询所有帖子
def get_all_posts():
    posts = graph.run("MATCH (p:Post) RETURN p").data()
    return posts


# 多表查询案例：查询所有优质用户，优质用户发布至少2个帖子，并且每个帖子的点赞数不少于1个
def get_all_excellent_users():
    excellent_users = graph.run("""
        MATCH (u:User)-[:POSTED]->(p:Post)
        WITH u, count(p) AS post_count
        WHERE post_count >= 1
        MATCH (u)-[:POSTED]->(p:Post)<-[:LIKED]-(u2:User)
        WITH u,
        count(DISTINCT p) AS liked_post_count,
        collect(DISTINCT p.id) AS liked_posts
        WHERE liked_post_count >= 1
        RETURN u.name AS username
    """).data()

    return excellent_users


# 复杂关系查询案例：查询所有优质用户的帖子，以及每个帖子的点赞用户
def get_all_excellent_users_posts():
    excellent_users_posts = graph.run("""
                                MATCH (u:User)-[:POSTED]->(p:Post)
                                WITH u, count(p) AS post_count
                                WHERE post_count >= 1
                                MATCH (u)-[:POSTED]->(p:Post)<-[:LIKED]-(u2:User)
                                WITH u,
                                count(DISTINCT p) AS liked_post_count,
                                collect(DISTINCT p.id) AS liked_posts
                                WHERE liked_post_count >= 1
                                UNWIND liked_posts AS post_id
                                MATCH (u)-[:POSTED]->(p:Post {id: post_id})
                                WITH u, p
                                MATCH (u2:User)-[:LIKED]->(p)
                                RETURN u, p, collect(DISTINCT
                                u2.name) AS liked_users
                                """).data()
    return excellent_users_posts


# 生成唯一的帖子ID
def generate_unique_comment_id():
    timestamp = int(time.time())
    random_number = random.randint(1000, 9999)
    comment_id = f"COMMENT-{timestamp}-{random_number}"
    return comment_id


def get_post_details(post_id):
    query = "MATCH (p:Post {id: $post_id})-[:POSTED_BY]->(u:User) RETURN p, u"
    result = graph.run(query, post_id=post_id).data()
    if result:
        post = result[0]['p']
        user = result[0]['u']
        print(post)
        return {'post': post, 'user': user}
    else:
        return None


# 获取优质用户的姓名
print([user["username"] for user in get_all_excellent_users()])
# 获取优质用户的帖子所有内容，以及每个帖子的点赞用户
print([(post["p"]["content"], post["liked_users"]) for post in get_all_excellent_users_posts()])
# 获取所有用户的姓名
print([user["u"]["name"] for user in get_all_users()])