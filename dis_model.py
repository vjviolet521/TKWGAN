from dis_modules import *


class Dis():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.item_count = itemnum
        self.user_count = usernum
        # self.h_size = args.h_size
        # self.v_size = args.v_size
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.label = tf.placeholder(tf.float32, shape=(None, 2))
        self.time_matrix = tf.placeholder(tf.int32, shape=(None, args.maxlen, args.maxlen))
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)
        self.time_matrix = tf.reshape(self.time_matrix, [tf.shape(self.input_seq)[0], args.maxlen, args.maxlen])
        with tf.variable_scope("discriminator", reuse=reuse):
            # -------------------------------------- ID ---------------------------------------
            # sequence embedding
            self.originseq = embedding(self.input_seq,
                                       vocab_size=itemnum + 1,
                                       num_units=args.hidden_units,
                                       zero_pad=True,
                                       scale=True,
                                       l2_reg=args.l2_emb,
                                       scope="input_embeddings_dis",
                                       with_t=True,
                                       reuse=reuse
                                       )
            self.Pu = embedding(self.u,
                                vocab_size=usernum + 1,
                                num_units=args.hidden_units,
                                zero_pad=True,
                                scale=True,
                                l2_reg=args.l2_emb,
                                scope="input_userembedding",
                                with_t=True,
                                reuse=reuse)

            absolute_pos_K = embedding(
              tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
              vocab_size=args.maxlen,
              num_units=args.hidden_units,
              zero_pad=False,
              scale=False,
              l2_reg=args.l2_emb,
              scope="abs_pos_K",
              reuse=reuse,
              with_t=False
            )
            absolute_pos_V = embedding(
              tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
              vocab_size=args.maxlen,
              num_units=args.hidden_units,
              zero_pad=False,
              scale=False,
              l2_reg=args.l2_emb,
              scope="abs_pos_V",
              reuse=reuse,
              with_t=False
            )

            # Time Encoding
            time_matrix_emb_K = embedding(
              self.time_matrix,
              vocab_size=args.time_span + 1,
              num_units=args.hidden_units,
              zero_pad=False,
              scale=False,
              l2_reg=args.l2_emb,
              scope="dec_time_K",
              reuse=reuse,
              with_t=False
            )
            time_matrix_emb_V = embedding(
              self.time_matrix,
              vocab_size=args.time_span + 1,
              num_units=args.hidden_units,
              zero_pad=False,
              scale=False,
              l2_reg=args.l2_emb,
              scope="dec_time_V",
              reuse=reuse,
              with_t=False
            )
            # Positional Encoding
            # t = embedding(
            #     tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
            #     vocab_size=args.maxlen,
            #     num_units=args.hidden_units,
            #     zero_pad=False,
            #     scale=False,
            #     l2_reg=args.l2_emb,
            #     scope="dec_pos_dis",
            #     reuse=reuse,
            #     with_t=True
            # )
            # self.seq += t
            # Dropout
            self.seq = tf.layers.dropout(self.originseq, rate=args.dis_dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            time_matrix_emb_K = tf.layers.dropout(time_matrix_emb_K,
                                                  rate=args.gen_dropout_rate,
                                                  training=tf.convert_to_tensor(self.is_training))
            time_matrix_emb_V = tf.layers.dropout(time_matrix_emb_V,
                                                  rate=args.gen_dropout_rate,
                                                  training=tf.convert_to_tensor(self.is_training))
            absolute_pos_K = tf.layers.dropout(absolute_pos_K,
                                               rate=args.gen_dropout_rate,
                                               training=tf.convert_to_tensor(self.is_training))
            absolute_pos_V = tf.layers.dropout(absolute_pos_V,
                                               rate=args.gen_dropout_rate,
                                               training=tf.convert_to_tensor(self.is_training))

            # Build blocks
            for i in range(args.dis_num_blocks):
                with tf.variable_scope("num_blocks_dis_%d" % i):

                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   time_matrix_K=time_matrix_emb_K,
                                                   time_matrix_V=time_matrix_emb_V,
                                                   absolute_pos_K=absolute_pos_K,
                                                   absolute_pos_V=absolute_pos_V,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.gen_num_heads,
                                                   dropout_rate=args.gen_dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention_dis",
                                                   )

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dis_dropout_rate, is_training=self.is_training)
                    self.seq *= mask
            self.seq = normalize(self.seq)
            self.seq = self.seq[:, -1, :]

            # -------------------------------------- KG ---------------------------------------
            # KG Embedding
            self.origin_kg_seq = kg_embedding(self.input_seq,
                                              num_units=args.hidden_units_kg,
                                              scope="kg_embeddings",
                                              reuse=reuse
                                              )

            # Positional Encoding
            t = embedding(
               tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
               vocab_size=args.maxlen,
               num_units=args.hidden_units_kg,
               zero_pad=False,
               scale=False,
               l2_reg=args.l2_emb,
               scope="dec_pos_dis_kg",
               reuse=reuse,
               with_t=True
            )
            # self.kg_seq += t

            self.kg_seq = self.originseq + t + self.origin_kg_seq

            avgout = tf.reduce_mean(self.kg_seq, axis=-1, keepdims=True)
            maxout = tf.math.reduce_max(self.kg_seq, axis=-1, keepdims=True)
            out = tf.concat([avgout, maxout], -1)
            # out = tf.transpose(out, perm=[0, 2, 1])
            out_conv = tf.layers.conv1d(out, filters=1, kernel_size=3, strides=1, padding='same')

            out_f = tf.math.sigmoid(out_conv)
            out_f = out_f * self.origin_kg_seq


            self.kg_seq = tf.layers.dropout(out_f, rate=args.dis_dropout_rate,
                                            training=tf.convert_to_tensor(self.is_training))

            self.kg_seq *= mask
            # Build blocks
            # for i in range(args.dis_num_blocks):
            #     with tf.variable_scope("num_blocks_dis_kg_%d" % i):
            #         # Self-attention
            #         self.kg_seq = multihead_attention_kg(queries=normalize(self.kg_seq),
            #                                             keys=self.kg_seq,
            #                                             num_units=args.hidden_units_kg,
            #                                             num_heads=args.dis_num_heads,
            #                                             dropout_rate=args.dis_dropout_rate,
            #                                             is_training=self.is_training,
            #                                             causality=False,
            #                                             scope="self_attention_dis_kg")
            #
            #         # Feed forward
            #         self.kg_seq = feedforward(normalize(self.kg_seq),
            #                                   num_units=[args.hidden_units_kg, args.hidden_units_kg],
            #                                   dropout_rate=args.dis_dropout_rate, is_training=self.is_training,
            #                                   scope="self_attention_dis_kg")
            #         self.kg_seq *= mask
            # self.kg_seq = normalize(self.kg_seq)
            # self.kg_seq = self.kg_seq[:, -1, :]

            self.kg_seq = caser_layer(msl=5,
                                      usr_seq=self.kg_seq,
                                      h_size=16,
                                      v_size=4,
                                      keep_prob=1.0,
                                      emb_size=50,
                                      Pu=self.Pu)

            # Final (unnormalized) scores and predictions
            l2_reg_lambda = 0.2
            l2_loss1 = tf.constant(0.0)
            with tf.name_scope("output1"):
                W1 = tf.Variable(tf.truncated_normal([args.hidden_units, 2], stddev=0.1), name="W1")
                b1 = tf.Variable(tf.constant(0.1, shape=[2]), name="b1")
                l2_loss1 += tf.nn.l2_loss(W1)
                l2_loss1 += tf.nn.l2_loss(b1)
                self.scores1 = tf.nn.xw_plus_b(self.seq, W1, b1, name="scores1")
                self.ypred_for_auc1 = tf.nn.softmax(self.scores1)

            l2_loss2 = tf.constant(0.0)
            with tf.name_scope("output2"):
                W2 = tf.Variable(tf.truncated_normal([args.hidden_units_kg, 2], stddev=0.1), name="W2")
                b2 = tf.Variable(tf.constant(0.1, shape=[2]), name="b2")
                l2_loss2 += tf.nn.l2_loss(W2)
                l2_loss2 += tf.nn.l2_loss(b2)
                self.scores2 = tf.nn.xw_plus_b(self.kg_seq, W2, b2, name="scores2")

                self.ypred_for_auc2 = tf.nn.softmax(self.scores2)

            self.ypred_for_auc = (self.ypred_for_auc1 + self.ypred_for_auc2 ) / 2

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores1, labels=self.label)
                loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores2, labels=self.label)

                self.loss = tf.reduce_mean(loss1 + loss2) + \
                            l2_reg_lambda * (l2_loss1 + l2_loss2)

        if reuse is None:
            self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
            self.global_step = tf.Variable(0, name='global_step_dis', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.dis_lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()


