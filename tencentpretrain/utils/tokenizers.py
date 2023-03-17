# -*- encoding:utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from tencentpretrain.utils.constants import *
from tencentpretrain.utils.vocab import Vocab
import collections
import unicodedata
import six
import regex as re


class Tokenizer(object):

    def __init__(self, args, is_src=True):
        self.vocab = None
        self.sp_model = None
        if is_src == True:
            spm_model_path = args.spm_model_path
            vocab_path = args.vocab_path
        else:
            spm_model_path = args.tgt_spm_model_path
            vocab_path = args.tgt_vocab_path

        if spm_model_path:
            try:
                import sentencepiece as spm
            except ImportError:
                raise ImportError("You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                                                    "pip install sentencepiece")
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(spm_model_path)
            self.vocab = {self.sp_model.IdToPiece(i): i for i
                                        in range(self.sp_model.GetPieceSize())}
        else:
            self.vocab = Vocab()
            self.vocab.load(vocab_path, is_quiet=True)
            self.vocab = self.vocab.w2i
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        if self.sp_model:
            return [self.sp_model.PieceToId(
                    printable_text(token)) for token in tokens]
        else:
            return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        if self.sp_model:
            return [self.sp_model.IdToPiece(id_) for id_ in ids]
        else:
            return convert_by_vocab(self.inv_vocab, ids)


class CharTokenizer(Tokenizer):

    def __init__(self, args, is_src=True):
        super().__init__(args, is_src)

    def tokenize(self, text, use_vocab=True):
        if use_vocab:
            return [token if token in self.vocab else UNK_TOKEN for token in list(text.strip())]
        else:
            return [token for token in list(text.strip())]


class SpaceTokenizer(Tokenizer):

    def __init__(self, args, is_src=True):
        super().__init__(args, is_src)

    def tokenize(self, text, use_vocab=True):
        if use_vocab:
            return [token if token in self.vocab else UNK_TOKEN for token in text.strip().split(" ")]
        else:
            return [token for token in text.strip().split(" ")]


SPIECE_UNDERLINE = u"▁".encode("utf-8")


def preprocess_text(inputs, remove_space=True, lower=False):
    """preprocess data by removing extra space and normalize data."""
    outputs = inputs
    if remove_space:
        outputs = " ".join(inputs.strip().split())

    if six.PY2 and isinstance(outputs, str):
        try:
            outputs = six.ensure_text(outputs, "utf-8")
        except UnicodeDecodeError:
            outputs = six.ensure_text(outputs, "latin-1")

    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    """turn sentences into word pieces."""

    if six.PY2 and isinstance(text, six.text_type):
        text = six.ensure_binary(text, "utf-8")

    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        piece = printable_text(piece)
        if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                    six.ensure_binary(piece[:-1]).replace(SPIECE_UNDERLINE, b""))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    # note(zhiliny): convert back to unicode for py2
    if six.PY2 and return_unicode:
        ret_pieces = []
        for piece in new_pieces:
            if isinstance(piece, str):
                piece = six.ensure_text(piece, "utf-8")
            ret_pieces.append(piece)
        new_pieces = ret_pieces

    return new_pieces


def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return six.ensure_text(text, "utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return six.ensure_text(text, "utf-8", "ignore")
        elif isinstance(text, six.text_type):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return six.ensure_text(text, "utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, six.text_type):
            return six.ensure_binary(text, "utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item] if item in vocab else vocab.get(UNK_TOKEN))
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.
    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BertTokenizer(Tokenizer):
    """Runs end-to-end tokenziation."""

    def __init__(self, args, is_src=True):
        super().__init__(args, is_src)
        if not args.spm_model_path:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=args.do_lower_case if is_src else args.tgt_do_lower_case)
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=UNK_TOKEN)

    def tokenize(self, text):
        if self.sp_model:
            split_tokens = encode_pieces(self.sp_model, text, return_unicode=False)
        else:
            split_tokens = []
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)

        return split_tokens


class BPETokenizer(Tokenizer):
    def __init__(self, args, is_src=True):
        super().__init__(args, is_src)

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(args.merges_path, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens


class XLMRobertaTokenizer(Tokenizer):
    """Runs end-to-end tokenziation."""

    def __init__(self, args, is_src=True):
        super().__init__(args, is_src)
        assert args.spm_model_path, \
            "spm_model_path must provided for huggingface roberta tokenizer"

        special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]
        vocab = [token for token in self.vocab if token not in special_tokens]
        vocab = special_tokens + vocab + ["<mask>"]
        self.vocab = {k: v for v, k in enumerate(vocab)}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        split_tokens = encode_pieces(self.sp_model, text, return_unicode=False)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):

        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):

        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case):
        """Constructs a BasicTokenizer.
        Args:
            do_lower_case: Whether to lower case the input.
        """
        if do_lower_case == "true":
            self.do_lower_case = True
        else:
            self.do_lower_case = False

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]
        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer.
        Returns:
            A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + six.ensure_str(substr)
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class ImageTokenizer(Tokenizer):
    """ Virtual tokenizer for vqgan models """

    def __init__(self, args, is_src=True):
        self.vocab = range(args.image_tokenizer["image_vocab_size"])


class VirtualTokenizer(Tokenizer):
    """ Virtual tokenizer for vit models """

    def __init__(self, args, is_src=True):
        self.vocab = []


class TextImageTokenizer(BertTokenizer):
    """ Text and image tokenizer (BERT and VQGAN) """

    def __init__(self, args, is_src=True):
        super().__init__(args, is_src)
        self.vocab_bias = len(self.vocab)
        for i in range(args.image_tokenizer["image_vocab_size"]):
            self.vocab[i + self.vocab_bias] = str(i)


from icetk.text_tokenizer import TextTokenizer
from icetk.utils import auto_create
import icetk.sentencepiece_model_pb2 as sp_model
class SPTokenizer:
    def __init__(
            self,
            vocab_file,
            max_blank_length=80,
            byte_fallback=True,
    ):
        assert vocab_file is not None
        self.vocab_file = vocab_file
        self.special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "<unused_0>", "<sop>", "<eop>", "<ENC>", "<dBLOCK>"]
        self.max_blank_length = max_blank_length
        self.byte_fallback = byte_fallback
        self.text_tokenizer = self._build_text_tokenizer(encode_special_tokens=False)
        self.special_text_tokenizer = self._build_text_tokenizer(encode_special_tokens=True)

    @staticmethod
    def _configure_tokenizer(
            text_tokenizer: TextTokenizer,
            special_tokens,
            max_blank_length: int,
            byte_fallback: bool,
            encode_special_tokens=False,
    ):
        # special token
        special_token_type = 4 if encode_special_tokens else 3  # 3 - CONTROL, 4 - USER_DEFINE
        for token in special_tokens:
            text_tokenizer.proto.pieces.append(
                sp_model.ModelProto.SentencePiece(piece=token, score=0.0, type=special_token_type)
            )
        # whitespaces
        for token in [SPTokenizer.get_tab_token()] + [
            SPTokenizer.get_blank_token(i) for i in range(2, max_blank_length + 1)
        ]:
            text_tokenizer.proto.pieces.append(sp_model.ModelProto.SentencePiece(piece=token, score=0.0, type=4))
        # byte fallback
        if byte_fallback:
            text_tokenizer.proto.trainer_spec.byte_fallback = True
            for i in range(256):
                text_tokenizer.proto.pieces.append(
                    sp_model.ModelProto.SentencePiece(piece="<0x{:02X}>".format(i), score=0.0, type=6)
                )
        text_tokenizer.refresh()

    def _build_text_tokenizer(self, encode_special_tokens=False):
        tokenizer = TextTokenizer(self.vocab_file)
        self._configure_tokenizer(
            tokenizer, self.special_tokens, self.max_blank_length, self.byte_fallback, encode_special_tokens
        )
        return tokenizer

    def _get_text_tokenizer(self, encode_special_tokens=False):
        if encode_special_tokens:
            return self.special_text_tokenizer
        else:
            return self.text_tokenizer

    @staticmethod
    def get_blank_token(length: int):
        assert length >= 2
        return f"<|blank_{length}|>"

    @staticmethod
    def get_tab_token():
        return f"<|tab|>"

    @property
    def num_image_tokens(self):
        return 20000

    @property
    def num_text_tokens(self):
        return self.text_tokenizer.num_tokens

    @property
    def num_tokens(self):
        return self.num_image_tokens + self.num_text_tokens

    @staticmethod
    def _encode_whitespaces(text: str, max_len: int = 80):
        text = text.replace("\t", SPTokenizer.get_tab_token())
        for i in range(max_len, 1, -1):
            text = text.replace(" " * i, SPTokenizer.get_blank_token(i))
        return text

    def _preprocess(self, text: str, linebreak=True, whitespaces=True):
        if linebreak:
            text = text.replace("\n", "<n>")
        if whitespaces:
            text = self._encode_whitespaces(text, max_len=self.max_blank_length)
        return text

    def encode(
            self, text: str, linebreak=True, whitespaces=True, special_tokens=False, add_dummy_prefix=True
    ):
        """
        @param text: Text to encode.
        @param linebreak: Whether to encode newline (\n) in text.
        @param whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
        @param special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
        @param add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self._preprocess(text, linebreak, whitespaces)
        if not add_dummy_prefix:
            text = "<n>" + text
        tmp = self._get_text_tokenizer(encode_special_tokens=special_tokens).encode(text)
        tokens = [x + self.num_image_tokens for x in tmp]
        return tokens if add_dummy_prefix else tokens[2:]

    def decode(self, text_ids, special_tokens=False) -> str:
        ids = [int(_id) - self.num_image_tokens for _id in text_ids]
        ids = [_id for _id in ids if _id >= 0]
        text = self._get_text_tokenizer(encode_special_tokens=special_tokens).decode(ids)
        text = text.replace("<n>", "\n")
        text = text.replace(SPTokenizer.get_tab_token(), "\t")
        for i in range(2, self.max_blank_length + 1):
            text = text.replace(self.get_blank_token(i), " " * i)
        return text

    def tokenize(
            self, text: str, linebreak=True, whitespaces=True, special_tokens=False, add_dummy_prefix=True
    ):
        """
        @param text: Text to encode.
        @param linebreak: Whether to encode newline (\n) in text.
        @param whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
        @param special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
        @param add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self._preprocess(text, linebreak, whitespaces)
        if not add_dummy_prefix:
            text = "<n>" + text
        tokens = self._get_text_tokenizer(encode_special_tokens=special_tokens).tokenize(text)
        return tokens if add_dummy_prefix else tokens[2:]

    def __getitem__(self, x):
        if isinstance(x, int):
            if x < self.num_image_tokens:
                return "<image_{}>".format(x)
            else:
                return self.text_tokenizer.convert_id_to_token(x - self.num_image_tokens)
        elif isinstance(x, str):
            if x.startswith("<image_") and x.endswith(">") and x[7:-1].isdigit():
                return int(x[7:-1])
            else:
                return self.text_tokenizer.convert_token_to_id(x) + self.num_image_tokens
        else:
            raise ValueError("The key should be str or int.")


class ChatGLMTokenizer(Tokenizer):
    """
    Construct a ChatGLM tokenizer. Based on byte-level Byte-Pair-Encoding.
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = {"vocab_file": "ice_text.model"}
    max_model_input_sizes = 2048
    model_input_names = ["input_ids"]

    def __init__(self, args, is_src=True):

        do_lower_case=False,
        remove_space=False,
        bos_token='sop',
        eos_token='eos',
        eop_token='eop',
        mask_token='[MASK]',
        gmask_token='[gMASK]',
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.vocab_file = args.vocab_path

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.eop_token = eop_token
        self.mask_token = mask_token
        self.gMASK_token = gmask_token

        self.sp_tokenizer = SPTokenizer(args.vocab_path)
        self.vocab = self.get_vocab()

        """ Initialisation """

    @property
    def eop_token_id(self):
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self.eop_token is None:
            return None
        return self.convert_tokens_to_ids(self.eop_token)

    @property
    def vocab_size(self):
        """ Returns vocab size """
        return self.sp_tokenizer.num_tokens

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs

        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def tokenize(self, text, **kwargs):
        """ Returns a tokenized string. """
        text = self.preprocess_text(text)

        seq = self.sp_tokenizer.tokenize(text)

        return seq

    def decode(
            self,
            token_ids,
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = True,
            spaces_between_special_tokens: bool = True,
            **kwargs
    ) -> str:
        if isinstance(token_ids[0], list):
            tokens = []
            for single_token_ids in token_ids:
                if self.pad_token_id in single_token_ids:  # remove pad
                    single_token_ids = list(filter((self.pad_token_id).__ne__, single_token_ids))
                tokens.append(self.sp_tokenizer.decode(single_token_ids))
            return (tokens)
        else:
            if self.pad_token_id in token_ids:  # remove pad
                token_ids = list(filter((self.pad_token_id).__ne__, token_ids))
            return self.sp_tokenizer.decode(token_ids)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.sp_tokenizer[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_tokenizer[index]

    def convert_tokens_to_ids(self, tokens):
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.convert_tokens_to_ids(i) for i in ids]

    def build_inputs_with_special_tokens(
            self, token_ids_0, token_ids_1 = None
    ):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is not None:
            token_ids_0 += token_ids_1
        mask_ids = self.sp_tokenizer[self.mask_token]
        gmask_ids = self.sp_tokenizer[self.gMASK_token]
        if mask_ids not in token_ids_0 and gmask_ids not in token_ids_0:
            token_ids_0 += [gmask_ids]

        if token_ids_0[-1] != mask_ids and token_ids_0[-1] != gmask_ids:
            token_ids_0 += [self.sp_tokenizer[self.eos_token]]

        token_ids_0 += [self.sp_tokenizer[self.bos_token]]

        return token_ids_0
