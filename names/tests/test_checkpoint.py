import unittest
import os
import shutil

from names import check, directories


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.check_path = directories.test_data("test_check")
        if os.path.isdir(self.check_path):
            shutil.rmtree(self.check_path),

    def tearDown(self):
        pass

    def test_clean(self):
        self.assertFalse(os.path.isdir(self.check_path))

    def test_checkdir_and_prefix(self):
        ch_default = check.CheckPoint()
        expected = os.path.abspath("training_checkpoints")
        actual = ch_default.checkpoint_dir
        self.assertEqual(expected, actual)
        expected = os.path.join(expected, "ckpt_{epoch}")
        actual = ch_default.checkpoint_prefix
        self.assertEqual(expected, actual)

        ch_specific = check.CheckPoint(
            base=self.check_path,
            checkpoint_dir="cool_training_checkpoints",
            prefix="cool_ckpt",
        )
        expected = os.path.join(self.check_path, "cool_training_checkpoints")
        actual = ch_specific.checkpoint_dir
        self.assertEqual(expected, actual)
        expected = os.path.join(expected, "cool_ckpt_{epoch}")
        actual = ch_specific.checkpoint_prefix
        self.assertEqual(expected, actual)

    def test_file_names(self):
        ch_specific = check.CheckPoint(
            base=self.check_path,
            checkpoint_dir="cool_training_checkpoints",
            prefix="cool_ckpt",
        )

        expected = os.path.join(self.check_path, "info.json")
        actual = ch_specific.info_path
        self.assertEqual(expected, actual)

        # model_path
        expected = os.path.join(self.check_path, "vocab.pkl")
        actual = ch_specific.vocab_path
        self.assertEqual(expected, actual)

        # vocab_path
        expected = os.path.join(self.check_path, "seq_model")
        actual = ch_specific.model_path
        self.assertEqual(expected, actual)

    def test_create_directories(self):
        ch_specific = check.CheckPoint(
            base=self.check_path,
            checkpoint_dir="cool_training_checkpoints",
            prefix="cool_ckpt",
        )
        ch_specific.create_dirs()

        self.assertTrue(os.path.isdir(self.check_path))
        self.assertTrue(os.path.isdir(ch_specific.checkpoint_dir))

    def test_save_info(self):
        ch_specific = check.CheckPoint(
            base=self.check_path,
            checkpoint_dir="cool_training_checkpoints",
            prefix="cool_ckpt",
        )
        ch_specific.save_info()
        path = os.path.join(ch_specific.base, "info.json")
        self.assertTrue(os.path.exists(path))

    def test_load_info(self):
        ch_specific = check.CheckPoint(
            base=self.check_path,
            checkpoint_dir="cool_training_checkpoints",
            prefix="cool_ckpt",
        )
        ch_specific.save_info()

        ch_reload = check.CheckPoint(base=self.check_path)
        expected = os.path.join(self.check_path, "cool_training_checkpoints")
        actual = ch_reload.checkpoint_dir
        self.assertEqual(expected, actual)
        expected = os.path.join(expected, "cool_ckpt_{epoch}")
        actual = ch_reload.checkpoint_prefix
        self.assertEqual(expected, actual)

    def test_default_vocab(self):
        expected = {
            "”",
            "v",
            "6",
            "`",
            "}",
            "\n",
            "ἱ",
            "ó",
            "ρ",
            "I",
            "≈",
            "M",
            "°",
            "O",
            "/",
            "H",
            "ἐ",
            "-",
            "ξ",
            "ô",
            "Y",
            "â",
            "1",
            "h",
            "ἰ",
            "9",
            "c",
            "ą",
            "m",
            "{",
            "=",
            "F",
            "ζ",
            "N",
            ":",
            "é",
            "]",
            "ώ",
            "ί",
            "ς",
            "Ä",
            "ὶ",
            "^",
            "d",
            "\\",
            "δ",
            "s",
            "í",
            "4",
            "ü",
            "z",
            "ὡ",
            "á",
            "ø",
            "…",
            "b",
            "L",
            "ú",
            "B",
            "S",
            "A",
            "ὰ",
            " ",
            "&",
            "Â",
            "<",
            "φ",
            "ā",
            "\t",
            "*",
            "5",
            "P",
            "!",
            "ι",
            "ö",
            "o",
            '"',
            "R",
            "χ",
            "k",
            "G",
            ".",
            "ī",
            ">",
            "ὅ",
            "W",
            "ì",
            "j",
            "―",
            "É",
            "û",
            "~",
            "ê",
            "È",
            "u",
            "+",
            "U",
            "2",
            "Ó",
            "$",
            "%",
            "K",
            "σ",
            "C",
            "γ",
            "[",
            "à",
            "w",
            "Ö",
            "Z",
            "p",
            "t",
            "ä",
            "–",
            "ł",
            "“",
            "X",
            "a",
            "Ω",
            "æ",
            "ˆ",
            "?",
            "(",
            "g",
            "D",
            "ã",
            "|",
            "—",
            "τ",
            "7",
            "è",
            "ν",
            "_",
            "x",
            "š",
            "’",
            "r",
            "y",
            "V",
            "'",
            "Õ",
            "Q",
            "i",
            "3",
            "α",
            ";",
            "#",
            "8",
            "f",
            "ο",
            "q",
            ")",
            "T",
            "ε",
            "e",
            "ί",
            "E",
            "™",
            "´",
            "υ",
            "π",
            "n",
            "@",
            ",",
            "ï",
            "ῦ",
            "0",
            "J",
            "l",
            "ë",
            "ç",
            "ἔ",
            "ό",
            "ñ",
            "ù",
            "λ",
            "η",
            "î",
        }
        actual = check.default_vocab()
        self.assertEqual(expected, actual)
        self.assertEqual(178, len(actual))