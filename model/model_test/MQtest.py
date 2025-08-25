# main()
#######################
# 需要初始化MQ的配置
MQ_config = {
    "group_name": "group",
    "name_server_address": "172.29.78.48:9876",
    "topic": "TopicTest"
}
#######################

# 初始化消费者
consumer = PushConsumer(MQ_config['group_name'])
# 使用 IP 和端口名称设置服务器地址
consumer.set_name_server_address(MQ_config['name_server_address'])
# consumer.set_session_credentials("access_key", "secret_key", 'authChannel')  # 完成设置验证
# 订阅Topic和过滤信息
consumer.subscribe(MQ_config['topic'], call_back, '*')

# 启动消费者, 此时会启动新的线程, 消费者将一直运行, 直到主进程被停止
consumer.start()
while True:
    time.sleep(3600)
# 停止消费者
consumer.shutdown()