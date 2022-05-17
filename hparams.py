import argparse

class HParams(object):
    def __init__(self):
        self.dataset_path = './dataset/gtzan'
        self.feature_path= './dataset/feature_augment'
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.app_genres = ['蓝调', '古典', '乡村', '迪斯科', '嘻哈', '爵士', '金属', '流行', '雷鬼', '摇滚']
        self.app_genres_discription = ['蓝调音乐始于美国南部。本为贫苦黑人劳动时呐喊的短曲，而其中亦混合了在教会中类似朗诵形式的节奏及韵律。\n'
                                       '蓝调音乐是黑人的思乡之曲。演唱或演奏时大量蓝调音的应用，使得音乐上充满了压抑及不和谐的感觉，这种音乐听起来十分忧郁。\n',
                                       '古典音乐有广义、狭义之分。广义是指那些从西方中世纪开始的、在欧洲主流文化背景下创作的西方古典音乐，主要因其复杂多样的创作技术和所能承载的厚重内涵而有别于通俗音乐和民间音乐。\n'
                                       '狭义指古典主义时期，1750年（J·S·巴赫去世）至1827年（贝多芬去世)，这一时期为古典主义音乐时期，它包含了两大时间段：“前古典时期”和“维也纳古典时期”。\n'
                                       '最为著名的维也纳乐派也是在“维也纳古典时期”兴起，其代表作曲家有海顿、莫扎特和贝多芬，被后世称为“维也纳三杰”。\n',
                                       '乡村音乐是一种具有美国民族特色的流行音乐，于20世纪20年代兴起于美国南部。\n'
                                       '其根源来自英国民谣，是美国白人民族音乐代表。乡村音乐的特点是曲调简单，节奏平稳，带有叙事性，具有较浓的乡土气息，亲切热情而不失流行元素。\n',
                                       '迪斯科是20世纪70年代初兴起的一种流行舞曲，电音曲风之一。音乐比较简单，具有强劲的节拍和强烈的动感。\n'
                                       '20世纪60年代中期，迪斯科传入美国，最初只是在纽约的一些黑人俱乐部里流传，20世纪70年代初逐渐发展成具有全国影响的一种音乐形式，并于20世纪70年代中期以后风靡世界。\n',
                                       '嘻哈，是一类流行文化（包括舞蹈、服饰、涂鸦等）的总称。诞生于美国贫民区街头的一种文化形式，首先在纽约市北部布朗克斯市区的非裔及拉丁裔青年之间兴起，继而发展壮大，并席卷全球。\n',
                                       '爵士乐，于19世纪末20世纪初源于美国，诞生于南部港口城市新奥尔良。\n'
                                       '与传统音乐比较而言，爵士乐的另一大特征是它的发音方法和音色，无论是乐器还是人声，这些特征都足以使人们绝不会将它们与任何传统音乐的音色混淆。爵士乐中的颤音是有变化的，变化的方向一般是幅度由窄到宽，速度由慢到快，而且常常在一个音临近结束时增加抖动的幅度和速度，更加强了这种技巧的表现力。同时，在一个音开始时，爵士乐手们会从下向上滑到预定的音高，在结束时，又从原来的音高滑下来。\n',
                                       '金属乐是广义摇滚乐下属的其中一种子风格，发源于二十世纪六十年代末至七十年代初的欧美地区，早年以英国和美国为关键发展阵地。\n'
                                       '金属乐主要根源于六十年代中后期已经发展成熟的迷幻摇滚和布鲁斯摇滚，以高失真的吉他连复段、沉重的贝斯线、密集或错落有致的鼓点、扭曲或悠扬高亢的吉他独奏，以及规整化、模块化的编曲结构为主要听觉特点。\n',
                                       '流行音乐准确的概念应为商品音乐，是指以赢利为主要目的而创作的音乐。流行音乐19世纪末20世纪初起源于美国。\n'
                                       '它的艺术性是次要的，市场性是主要的。从音乐体系看，流行音乐是在布鲁斯、爵士乐、摇滚乐、等美国大众音乐架构基础上发展起来的音乐。其风格多样，形态丰富，可泛指Jazz、Rock、Soul、Blues、Reggae、Rap、Hip-Hop、Disco、New Age等20世纪后诞生的都市化大众音乐。\n'
                                       '中国流行音乐的风格与形态主要受欧美影响，在此基础上逐渐形成本土风格。\n',
                                       '雷鬼音乐结合了传统非洲节奏，美国的节奏蓝调及原始牙买加民俗音乐， 这种风格包含了”弱拍中音节省略”，“向上拍击的吉他弹奏”， 以及”人声合唱”。 歌词强调社会、政治及人文的关怀。\n'
                                       '雷鬼乐是早期牙买加的流行音乐之一，它不仅融合了美国节奏蓝调的抒情曲风，同时还加入了拉丁音乐的热情。另外，雷鬼乐十分强调vocal的部份，不论是独唱或合唱，通常它是运用吟唱的方式来表现，并且藉由吉他、打击乐器、电子琴或其他乐器带出主要的旋律和节奏。\n',
                                       '摇滚，起源于20世纪40年代末期的美国，20世纪50年代早期开始流行，迅速风靡全球。摇滚乐以其灵活大胆的表现形式和富有激情的音乐节奏表达情感，受到了全世界大多数人的喜爱，并在1960年和1970年形成一股热潮。\n'
                                       '摇滚主要受到布鲁斯、乡村音乐等影响发展而来。早期摇滚乐很多都是黑人布鲁斯的翻唱版，因而布鲁斯是其主要根基。\n'
                                       '摇滚乐分支众多，形态复杂，主要风格有：民谣摇滚、艺术摇滚、迷幻摇滚、乡村摇滚、重金属、朋克、另类摇滚等。\n'
                                       ]

        # Feature Parameters
        self.sample_rate = 22050
        self.fft_size = 1024
        self.win_size = 1024
        self.hop_size = 512
        self.num_mels = 128
        self.feature_length = 1024

        # Training Parameters
        self.device = 1  # 0: CPU, 1: GPU0, 2: GPU1, ...
        self.batch_size = 8
        self.num_epochs = 70
        self.learning_rate = 1e-2
        # self.learning_rate = 0.1
        self.stopping_rate = 1e-5
        # self.weight_decay = 1e-6
        self.weight_decay = 0.001
        self.momentum = 0.9
        self.factor = 0.2
        self.patience = 5

        # self.model_path = './standard_resnet_model.pkl'
        # self.model_path = './82.88resnet50_pretrained.pkl'
        self.model_path = './resnet_3channel(no_aug).pkl'


    # Function for pasing argument and set hParams
    def parse_argument(self, print_argument=True):
        parser = argparse.ArgumentParser()
        for var in vars(self):
            value = getattr(hparams, var)
            argument = '--' + var
            parser.add_argument(argument, type=type(value), default=value)

        args = parser.parse_args()
        for var in vars(self):
            setattr(hparams, var, getattr(args,var))

        if print_argument:
            print('----------------------')
            print('Hyper Paarameter Settings')
            print('----------------------')
            for var in vars(self):
                value = getattr(hparams, var)
                print(var + ":" + str(value))
            print('----------------------')

hparams = HParams()
hparams.parse_argument()
