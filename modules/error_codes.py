from __future__ import annotations

"""
Единая система ошибок с кодами и человекочитаемыми сообщениями.

Формат выдачи: "Ошибка [<CODE>]: <Краткое описание>. Этап: <stage>. Детали: <detail>"

Использование:
- raise AppError('E002', 'DB.verify', 'read of closed file')
- msg = format_error('E006', 'Update.meta', 'version.json signature invalid')
"""

from dataclasses import dataclass
from typing import Optional


ERROR_TITLES: dict[str, str] = {
    'E001': 'Ошибка подключения к БД',
    'E002': 'Ошибка запроса к БД',
    'E003': 'Неверный формат ID',
    'E004': 'Неверный пароль',
    'E005': 'Пользователь не найден',
    'E006': 'Не удалось верифицировать метаданные обновления',
    'E007': 'Ключ подписи обновлений отсутствует',
    'E008': 'Некорректные аргументы при отправке файла',
    'E009': 'Ошибка записи файла',
    'E010': 'Несовпадение размеров файла',
    'E011': 'Недостаточно прав для комнаты',
    'E012': 'Требуется аутентификация',
    # Общие для авторизации/регистрации/валидации
    'E013': 'Отсутствует ID пользователя',
    'E014': 'Пароль пустой',
    'E015': 'Пароль слишком короткий',
    'E016': 'Пароль слишком длинный',
    'E017': 'Слишком много попыток',
    'E018': 'Пользователь уже онлайн',
    'E019': 'Внутренняя ошибка сервера',
}


def format_error(code: str, stage: str, detail: Optional[str] = None) -> str:
    title = ERROR_TITLES.get(code, 'Неизвестная ошибка')
    stage = (stage or '').strip() or '—'
    detail = (detail or '').strip()
    base = f"Ошибка [{code}]: {title}. Этап: {stage}."
    if detail:
        return f"{base} Детали: {detail}"
    return base


@dataclass
class AppError(Exception):
    code: str
    stage: str
    detail: Optional[str] = None

    def __str__(self) -> str:
        return format_error(self.code, self.stage, self.detail)
