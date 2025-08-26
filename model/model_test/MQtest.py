import time
from rocketmq.client import PushConsumer  # 确保导入PushConsumer


def call_back(msg):
    """消息回调函数示例，需要根据实际需求实现"""
    print(f"Received message: {msg.body}")
    # 处理消息的逻辑
    return ConsumeStatus.CONSUME_SUCCESS  # 假设返回消费状态


def main():
    #######################
    # 需要初始化MQ的配置
    MQ_config = {
        "group_name": "group",
        "name_server_address": "172.29.78.48:9876",
        "topic": "TopicTest"
    }
    #######################

    try:
        # 初始化消费者
        consumer = PushConsumer(MQ_config['group_name'])
        # 使用 IP 和端口名称设置服务器地址
        consumer.set_name_server_address(MQ_config['name_server_address'])
        # consumer.set_session_credentials("access_key", "secret_key", 'authChannel')  # 完成设置验证

        # 订阅Topic和过滤信息
        consumer.subscribe(MQ_config['topic'], call_back, '*')

        # 启动消费者, 此时会启动新的线程, 消费者将一直运行, 直到主进程被停止
        consumer.start()
        print("Consumer started. Waiting for messages...")

        # 保持主线程运行
        while True:
            time.sleep(3600)

    except KeyboardInterrupt:
        print("Shutting down consumer...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 确保消费者被正确关闭
        consumer.shutdown()
        print("Consumer shutdown complete.")


if __name__ == "__main__":
    main()