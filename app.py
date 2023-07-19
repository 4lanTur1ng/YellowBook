import neo
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from py2neo import Graph, Node, Relationship
from neo import create_user, get_user_id, create_post, check_password, update_password, create_comment, create_like
from extfunc import generate_unique_comment_id, get_post_details
import datetime
import nlp

graph = Graph("bolt://localhost:7687", user="neo4j", password="12345678")

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# 加载模型
# model = torch.jit.load('model.th')

@app.route('/create_user', methods=['GET', 'POST'])
def create_new_user():
    if request.method == 'POST':
        username = request.form.get('username')
        age = request.form.get('age')
        password = request.form.get('password')

        # 查询用户名是否已存在
        query = "MATCH (u:User) WHERE u.name = $username RETURN u"
        result = graph.run(query, username=username).data()

        if result:
            # 用户名已存在，返回错误信息并跳回首页
            return redirect(url_for('err_exists'))

        # 查询当前所有用户中的最大id
        query = "MATCH (u:User) RETURN max(u.id) AS max_id"
        result = graph.run(query).data()
        max_id = result[0]['max_id'] if (result and result[0]['max_id'] is not None) else 0

        # 创建新用户的id为最大id + 1
        user_id = max_id + 1

        # 执行创建新用户的逻辑
        create_user(user_id, username, age, password)
        session['username'] = username
        session['user_id'] = user_id
        # 重定向到新的欢迎页面
        return redirect(url_for('welcome_user', username=username))

    # 处理GET请求的逻辑
    return render_template('create_user.html')


@app.route('/welcome/<username>')
def welcome_user(username):
    # 查询当前用户创建的帖子及其评论和点赞数
    query = """
        MATCH (u:User {name: $username})-[:POSTED]->(p:Post)
        OPTIONAL MATCH (p)<-[:COMMENTED]-(c:Comment)
        OPTIONAL MATCH (p)<-[:LIKED_POST]-(liker:User)
        RETURN p, COUNT(DISTINCT c) AS comment_count, COUNT(DISTINCT liker) AS likes_count
        ORDER BY p.timestamp DESC
    """
    posts_result = graph.run(query, username=username).data()
    posts = [
        {
            'post': post['p'],
            'comment_count': post['comment_count'],
            'likes_count': post['likes_count']
        }
        for post in posts_result
    ]

    return render_template('welcome.html', username=username, posts=posts)



@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return render_template('user_profile.html', username=username)


@app.route('/high_quality_users')
def high_quality_users():
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

    users = [user["username"] for user in excellent_users]

    return render_template('high_quality_users.html', users=users)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sign_in', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # 查询用户是否存在
        query = "MATCH (u:User) WHERE u.name = $username RETURN u"
        result = graph.run(query, username=username).data()

        if result:
            user_node = result[0]['u']
            # 检查密码是否正确
            if check_password(user_node, password):
                # 密码正确，登录成功，跳转到欢迎页面
                # 获取用户的user_id
                user_id = user_node['id']

                # 存储username和user_id到session中
                session['username'] = username
                session['user_id'] = user_id
                return redirect(url_for('welcome_user', username=username))
            else:
                # 密码错误，登录失败，返回错误信息
                return "Incorrect password", 400
        else:
            # 用户不存在，登录失败，返回错误信息
            return "User does not exist", 400

    # 处理GET请求的逻辑，显示登录表单
    return render_template('sign_in.html')


@app.route('/post/<post_id>')
def show_post(post_id):
    # 查询帖子的详细信息和所有关联的评论
    post = graph.run("MATCH (p:Post) WHERE p.id = $post_id RETURN p", post_id=post_id).data()
    print(post)
    if not post:  # 如果没有找到对应的帖子
        return "Post not found", 404  # 可以返回一个404错误，或者重定向到其他页面

    comments = graph.run("MATCH (p:Post)<-[:COMMENTED]-(c:Comment) WHERE p.id = $post_id RETURN c",
                         post_id=post_id).data()

    return render_template('post_detail.html', post=post[0], comments=comments)


@app.route('/err_exists')
def err_exists():
    # 渲染错误页面
    return render_template('err_exists.html')


@app.route('/favicon.ico')
def favicon():
    return ''


@app.route('/user_details/<username>', methods=['GET', 'POST'])
def user_details(username):
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')

        # 获取用户节点
        user_node = graph.nodes.match("User", name=username).first()

        # 验证当前密码是否正确
        if not check_password(user_node, current_password):
            return "Current password is incorrect."

        # 更新密码
        update_password(user_node, new_password)

        return "Password updated successfully."

    # 处理GET请求的逻辑
    user_node = graph.nodes.match("User", name=username).first()
    print(user_node)
    return render_template('user_details.html', username=username, user=user_node)


# @app.route('/hall')
# # def hall():
# #     # 查询未封禁用户所发的帖子及其用户名
# #     query = """
# #         MATCH (u:User)-[:POSTED]->(p:Post)
# #         WHERE NOT u.banned
# #         RETURN p, u.name AS username
# #     """
# #     result = graph.run(query).data()
# #     posts = [{'post': record['p'], 'username': record['username']} for record in result]
# #     print(posts)
# #     return render_template('hall.html', posts=posts)


# 定义可选的分区列表
label = ['DS', 'SEKIRO', 'genshin-impact', 'MGS', 'MH', 'star-rail']    # 游戏标签


@app.route('/hall')
def hall():
    # 查询未封禁用户所发的帖子及其用户名，并按分区进行分组
    query = """
        MATCH (u:User)-[:POSTED]->(p:Post)
        WHERE NOT u.banned
        RETURN p, u.name AS username, p.partition AS partition
    """
    result = graph.run(query).data()

    # 创建字典以存储按分区分组的帖子列表
    grouped_posts = {partition: [] for partition in label}

    # 将帖子记录分组存储到对应的分区列表中
    for record in result:
        post = {'post': record['p'], 'username': record['username']}
        partition = record['partition']
        grouped_posts[partition].append(post)

    return render_template('hall.html', grouped_posts=grouped_posts)




@app.route('/create_post', methods=['POST'])
def create_post():

    post_content = request.form.get('post_content')  # 获取表单中的帖子内容
    user_id = session.get('user_id')  # 从session中获取当前登录的用户id
    partition = label[int(nlp.predict(post_content))] # to do nlp获取表单中的帖子分区

    # 查询当前所有帖子中的最大id
    query = "MATCH (p:Post) RETURN max(p.id) AS max_id"
    result = graph.run(query).data()
    max_id = result[0]['max_id'] if result and result[0]['max_id'] is not None else 0

    # 创建新帖子的id为最大id + 1
    post_id = max_id + 1
    # 获取当前时间作为发表时间
    timestamp = datetime.datetime.now()
    # 调用创建新帖子的函数
    neo.create_post(post_id, post_content, user_id, partition, timestamp=timestamp)

    return redirect(url_for('welcome_user', username=session['username']))  # 重定向回欢迎页面    return redirect(url_for('welcome_user', username=session['username']))  # 重定向回欢迎页面


@app.route('/post_details/<post_id>', methods=['GET', 'POST'])
def post_details(post_id):
    if request.method == 'POST':
        comment_content = request.form.get('comment_content')
        user_id = session.get('user_id')
        comment_id = generate_unique_comment_id()
        # 获取当前时间作为发表时间
        timestamp = datetime.datetime.now()
        neo.create_comment(comment_id, comment_content, post_id, user_id, timestamp=timestamp)
        return redirect(url_for('post_details', post_id=post_id))

    tmp = int(post_id)
    post_query = """
        MATCH (p:Post {id: $post_id})
        OPTIONAL MATCH (p)<-[:LIKED_POST]-(liker:User)
        RETURN p, COUNT(liker) AS likes_count
    """
    post_result = graph.run(post_query, post_id=tmp).data()
    post = post_result[0]['p']
    likes_count = post_result[0]['likes_count']
    post_result = graph.run(post_query, post_id=tmp).data()
    post = post_result[0]['p']

    comments_query = """
        MATCH (p:Post {id: $post_id})<-[:COMMENTED]-(c:Comment)<-[:CREATED]-(u:User)
        OPTIONAL MATCH (c)<-[:LIKED_COMMENT]-(liker:User)
        RETURN c, u, COUNT(liker) AS likes_count
    """
    comments_result = graph.run(comments_query, post_id=tmp).data()
    comments = [
        {
            'comment': comment['c'],
            'user': comment['u'],
            'likes': comment['likes_count']
        }
        for comment in comments_result
    ]
    print(comments)

    return render_template('post_details.html', post=post, likes_count=likes_count, comments=comments)



@app.route('/create_comment/<post_id>', methods=['POST'])
def create_comment(post_id):
    comment_content = request.form.get('comment_content')
    user_id = session.get('user_id')
    # 生成唯一的评论ID，可以使用您自己的方法
    comment_id = generate_unique_comment_id()
    # 获取当前时间作为发表时间
    timestamp = datetime.datetime.now()
    neo.create_comment(comment_id, comment_content, post_id, user_id, timestamp=timestamp, abandoned=False)
    return redirect(url_for('post_details', post_id=post_id))


@app.route('/create_like', methods=['POST'])
def create_like():
    user_id = session.get('user_id')
    post_id = request.form.get('post_id')
    comment_id = request.form.get('comment_id')
    print(post_id)
    if post_id:
        neo.create_like(post_id=post_id, user_id=user_id)
        return redirect(url_for('post_details', post_id=post_id))
    elif comment_id:
        neo.create_like(comment_id=comment_id, user_id=user_id)
        post_id = neo.get_post_id_from_comment_id(comment_id)  # Get the post_id associated with the comment_id
        return redirect(url_for('post_details', post_id=post_id))

    return redirect(url_for('hall'))  # Redirect to the hall page if neither post_id nor comment_id is provided

@app.route('/delete_post', methods=['POST'])
def delete_post():
    post_id = request.form.get('post_id')

    if post_id is not None and post_id.isdigit():
        # 在 Neo4j 数据库中查找帖子节点
        query = "MATCH (p:Post) WHERE p.id = $post_id RETURN p"
        result = graph.run(query, post_id=int(post_id)).data()

        if result:
            post_node = result[0]['p']
            # 删除帖子节点及其关系
            graph.delete(post_node)

            return jsonify({'message': 'Post deleted successfully'})
        else:
            return jsonify({'message': 'Post does not exist or has already been deleted'})
    else:
        return jsonify({'message': 'Invalid post ID'})

if __name__ == '__main__':
    app.run(debug=True)

