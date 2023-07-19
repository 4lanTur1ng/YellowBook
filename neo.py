from flask import redirect, url_for
from py2neo import Graph, Node, Relationship
import datetime
import nlp


# 连接到Neo4j数据库
graph = Graph("bolt://localhost:7687", user="neo4j", password="12345678")


# 创建用户节点，增加封禁状态和密码属性
def create_user(user_id, username, age, password, is_banned=False):
    user_node = Node("User", id=user_id, name=username, age=age, password=password, banned=is_banned)
    graph.create(user_node)


def get_user_id(username):
    query = "MATCH (u:User {name: $username}) RETURN u.id AS id"
    result = graph.run(query, username=username).data()
    if result:
        return result[0]['id']
    else:
        return None


# 创建帖子节点，添加时间属性和abandoned属性，包含分区属性为枚举类型，同时检查用户是否被封禁
def create_post(post_id, content, user_id, partition, timestamp=None, abandoned=False):
    user_node = graph.nodes.match("User", id=user_id).first()
    # 检查用户是否被封禁
    if user_node["banned"]:
        print("This user is banned and cannot post.")
        return
    cypher = '''
    MATCH (u:User)
    WHERE u.id = $user_id
    CREATE (p:Post {id: $post_id, content: $content, user_id: $user_id, partition: $partition, abandoned: $abandoned, timestamp: $timestamp})
    CREATE (u)-[:POSTED]->(p)
    '''
    graph.run(cypher, user_id=user_id, post_id=post_id, content=content, partition=partition, abandoned=abandoned, timestamp=timestamp)


# 创建评论节点，增加时间属性和abandoned属性，同时检查用户是否被封禁
def create_comment(comment_id, content, post_id, user_id, timestamp=None, abandoned=False):
    user_node = graph.nodes.match("User", id=user_id).first()
    # 检查用户是否被封禁
    if user_node["banned"]:
        print("This user is banned and cannot post comments.")
        return
    comment_node = Node("Comment", id=comment_id, content=content, abandoned=abandoned, timestamp=timestamp)
    tmp = int(post_id)
    post_node = graph.run("MATCH (p:Post) WHERE p.id = $post_id RETURN p", post_id=tmp).data()
    print(post_node)
    post_node = post_node[0]['p']
    commented_rel = Relationship(comment_node, "COMMENTED", post_node)
    created_rel = Relationship(user_node, "CREATED", comment_node)
    graph.create(commented_rel)
    graph.create(created_rel)


# 创建点赞关系
def create_like(post_id=None, comment_id=None, user_id=None):
    if post_id:
        # print(post_id)
        post_node = graph.nodes.match("Post", id=int(post_id)).first()
        print(post_node)
        user_node = graph.nodes.match("User", id=user_id).first()
        liked_rel = Relationship(user_node, "LIKED_POST", post_node)
        graph.create(liked_rel)
    elif comment_id:
        comment_node = graph.nodes.match("Comment", id=comment_id).first()
        user_node = graph.nodes.match("User", id=user_id).first()
        liked_rel = Relationship(user_node, "LIKED_COMMENT", comment_node)
        graph.create(liked_rel)


# 编辑用户属性
def edit_user(user_id, new_username):
    user_node = graph.nodes.match("User", id=user_id).first()
    user_node["name"] = new_username
    graph.push(user_node)


# 创建审核节点
def create_moderator(moderator_id, name):
    moderator_node = Node("Moderator", id=moderator_id, name=name)
    graph.create(moderator_node)


# 创建删除评论的权限关系
def grant_delete_comment_permission(moderator_id, comment_id):
    moderator_node = graph.nodes.match("Moderator", id=moderator_id).first()
    comment_node = graph.nodes.match("Comment", id=comment_id).first()
    permission_rel = Relationship(moderator_node, "CAN_DELETE", comment_node)
    graph.create(permission_rel)


# 更改用户封禁状态
def change_user_ban_status(user_id, is_banned):
    user_node = graph.nodes.match("User", id=user_id).first()
    user_node["banned"] = is_banned
    graph.push(user_node)


# 创建删除帖子的权限关系
def grant_delete_post_permission(moderator_id, post_id):
    moderator_node = graph.nodes.match("Moderator", id=moderator_id).first()
    post_node = graph.nodes.match("Post", id=post_id).first()
    permission_rel = Relationship(moderator_node, "CAN_DELETE", post_node)
    graph.create(permission_rel)


# 伪删除评论
def pseudo_delete_comment(moderator_id, comment_id):
    # 检查审核员是否有删除这个评论的权限
    moderator_node = graph.nodes.match("Moderator", id=moderator_id).first()
    comment_node = graph.nodes.match("Comment", id=comment_id).first()

    if not moderator_node or not comment_node:
        print("Either the moderator or the comment does not exist.")
        return

    permission_rel = graph.match_one(nodes=(moderator_node, comment_node), r_type="CAN_DELETE")

    if permission_rel is None:
        print("The moderator does not have the permission to delete this comment.")
        return

    # 如果审核员有删除这个评论的权限，那么修改评论属性并删除与其他节点的关系
    comment_node["abandoned"] = True
    graph.push(comment_node)

    graph.run("MATCH (c:Comment {id: $comment_id})-[r]-() DELETE r", comment_id=comment_id)
    print("Comment has been marked as abandoned.")


# 伪删除帖子
def pseudo_delete_post(moderator_id, post_id):
    # 检查审核员是否有删除这个帖子的权限
    moderator_node = graph.nodes.match("Moderator", id=moderator_id).first()
    post_node = graph.nodes.match("Post", id=post_id).first()

    if not moderator_node or not post_node:
        print("Either the moderator or the post does not exist.")
        return

    permission_rel = graph.match_one(nodes=(moderator_node, post_node), r_type="CAN_DELETE")

    if permission_rel is None:
        print("The moderator does not have the permission to delete this post.")
        return

    # 如果审核员有删除这个帖子的权限，那么修改帖子属性并删除与其他节点的关系
    post_node["abandoned"] = True
    graph.push(post_node)

    graph.run("MATCH (p:Post {id: $post_id})-[r]-() DELETE r", post_id=post_id)
    print("Post has been marked as abandoned.")


# 检查密码是否正确
def check_password(user_node, password):
    # 在这里实现密码验证逻辑
    # 比较用户节点中存储的密码与输入的密码是否匹配
    stored_password = user_node['password']
    if stored_password == password:
        return True
    else:
        return False


# 更新密码
def update_password(user_node, new_password):
    # 更新密码的逻辑
    user_node['password'] = new_password
    graph.push(user_node)


# 获取用户的帖子对应的评论
def get_post_id_from_comment_id(comment_id):
    query = "MATCH (:Comment {id: $comment_id})-[:COMMENTED]->(post:Post) RETURN post.id AS post_id"
    result = graph.run(query, comment_id=comment_id).data()
    if result:
        return result[0]['post_id']
    else:
        return None


# 枚举所有分区
tags = ["Study", "Life", "Work", "Others"]


# create_like(post_id=1, user_id=1)
# create_like(comment_id='COMMENT-1689324134-5030', user_id=1)